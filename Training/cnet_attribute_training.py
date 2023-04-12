import argparse
import MinkowskiEngine as ME
from torch.optim import lr_scheduler
from sklearn.metrics import f1_score
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers
import torch
import torch.nn as nn
import numpy as np

from DataPreprocessing.training_data_pipeline import data_collector

'''
In this implementation we use pytorch lightning for training on multiple clusters
The model architecture is constructed in the class ColorVoxelDNNTrainModule()
Training is setup in the class train_MoL_ColorVoxelDNN()
to start the training, run: 

python3 -m Training.cnet_attribute_training -trainset ../Datasets/CNeT_TrainingSet/33K/  -validset ../Datasets/CNeT_TrainingSet/33K/ -flag test -outputmodel Model/  -lr 7 -useDA 8   --color -opt 1 -dim 2 -ngpus 1  -batch 2  -bacc 1

check all the arguments at the end of this file

'''
class maskedConv3D(ME.MinkowskiConvolution):
    def __init__(self, masktype, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', self.kernel.data.clone())

        kD, _, _ = self.kernel.size()

        self.mask.fill_(1)
        self.mask[kD // 2 + (masktype == 'B'):, :, :] = 0

    def forward(self, x):
        self.kernel.data *= self.mask
        return super(maskedConv3D, self).forward(x)


class residualBlock(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.no_filter = h
        self.spconva = ME.MinkowskiConvolution(in_channels=2 * h, out_channels=h, kernel_size=1, dimension=3)
        self.spconvb = maskedConv3D(masktype='B', in_channels=h, out_channels=h, kernel_size=5, dimension=3)
        self.spconvc = ME.MinkowskiConvolution(in_channels=h, out_channels=2 * h, kernel_size=1, dimension=3)
        self.resnetblock = nn.Sequential(
            ME.MinkowskiELU(),
            self.spconva,
            ME.MinkowskiELU(),
            self.spconvb,
            ME.MinkowskiELU(),
            self.spconvc,
        )


    def forward(self, x,y=None):
        if(y is None):
            identity = x
            out = self.resnetblock(x)
            out += identity
        else:
            identity = x
            y = ME.SparseTensor(features=y.F, coordinate_manager=x.coordinate_manager,
                                coordinate_map_key=x.coordinate_map_key)
            out=x+y
            out = self.resnetblock(out)
            out += identity
        return out



class residualBlock_nomask(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.no_filter = h
        self.spconva = ME.MinkowskiConvolution(in_channels=2 * h, out_channels=h, kernel_size=1, dimension=3)
        self.spconvb = ME.MinkowskiConvolution( in_channels=h, out_channels=h, kernel_size=5, dimension=3)
        self.spconvc = ME.MinkowskiConvolution(in_channels=h, out_channels=2 * h, kernel_size=1, dimension=3)
        self.resnetblock = nn.Sequential(
            ME.MinkowskiELU(),
            self.spconva,
            ME.MinkowskiELU(),
            self.spconvb,
            ME.MinkowskiELU(),
            self.spconvc,
        )

    def forward(self, x):
        identity = x
        out = self.resnetblock(x)
        out += identity
        return out



class ResidualColorVoxelDNN(nn.Module):
    def __init__(self, input_channels, no_res, dimension):
        nn.Module.__init__(self)
        self.input_channels = input_channels
        self.no_residual = no_res
        self.softmax=nn.Softmax(dim=1)
        if dimension !=0:
            self.dim=dimension
        else:
            self.dim=3
        self.color_voxeldnn = nn.Sequential(
            maskedConv3D(masktype='A', in_channels=self.input_channels, out_channels=64, kernel_size=7, dimension=3),
            *[residualBlock(32) for _ in range(self.no_residual)],

            ME.MinkowskiELU(),
            maskedConv3D(masktype='B',in_channels=64, out_channels=128, kernel_size=3,dimension=3),
            *[residualBlock(64) for _ in range(self.no_residual)],

            ME.MinkowskiELU(),
            maskedConv3D(masktype='B',in_channels=128, out_channels=256, kernel_size=3, dimension=3),
            *[residualBlock(128) for _ in range(self.no_residual)],

            ME.MinkowskiELU(),
            maskedConv3D(masktype='B', in_channels=256, out_channels=512, kernel_size=3, dimension=3),
            *[residualBlock(256) for _ in range(self.no_residual+2)]

        )
        self.tail=nn.Sequential(
            ME.MinkowskiELU(),
            *[residualBlock(256) for _ in range(self.no_residual)],
        )
        self.channelcondition=nn.Sequential(
            ME.MinkowskiConvolution( in_channels=self.dim, out_channels=512, kernel_size=7, dimension=3),
            ME.MinkowskiELU(),
            *[residualBlock_nomask(256) for _ in range(self.no_residual)],
        )
        self.chroma_last=nn.Sequential(
            ME.MinkowskiConvolution( in_channels=512, out_channels=512, kernel_size=1, dimension=3),
        )
        self.luma_last = nn.Sequential(
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(in_channels=512, out_channels=256, kernel_size=1, dimension=3),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(in_channels=256, out_channels=256, kernel_size=1, dimension=3),
        )

    def forward(self, x,y):
        x1 = self.color_voxeldnn(x)
        if self.dim!=3:
            y1=self.channelcondition(y)
            y1 = ME.SparseTensor(features=y1.F, coordinate_manager=x1.coordinate_manager,
                                coordinate_map_key=x1.coordinate_map_key)
            out=x1+y1
            out = self.tail(out)
            out = out + y1
            out=self.chroma_last(out)
        else:
            out = self.tail(x1)
            out=self.luma_last(out)
        return out.F

def loss_op(loss,true_input, prediction):
    true_input = true_input.long()
    output = loss(prediction, true_input)
    return output

def compute_metrics(predicts, true_input):
    sampled_probs = predicts.detach().clone()
    sampled_probs = sampled_probs.cpu()
    true_input =  torch.round(true_input.cpu())

    _, t1 = torch.topk(sampled_probs, 1, 1, True, True)
    correct_t1 = torch.eq(true_input[:, None], t1)
    Top1_accuracy = correct_t1.float().mean()

    fmarco = f1_score(true_input, t1, average='macro')
    fmicro = f1_score(true_input, t1, average='micro')
    fweight = f1_score(true_input, t1, average='weighted')

    _, t3 = torch.topk(sampled_probs, 3, 1, True, True)
    correct_t3 = torch.eq(true_input[:, None], t3).any(dim=1)
    Top3_accuracy = correct_t3.float().mean()

    _, t5 = torch.topk(sampled_probs, 5, 1, True, True)
    correct_t5 = torch.eq(true_input[:, None], t5).any(dim=1)
    Top5_accuracy = correct_t5.float().mean()

    _, t20 = torch.topk(sampled_probs, 20, 1, True, True)
    correct_t20 = torch.eq(true_input[:, None], t20).any(dim=1)
    Top20_accuracy = correct_t20.float().mean()
    return Top1_accuracy,Top3_accuracy, Top5_accuracy,Top20_accuracy, fmarco, fmicro,fweight

def train_MoL_ColorVoxelDNN(args):
    params = {'batch_size': args.batch,
              'shuffle': True,
              'num_workers': 4}


    training_generator, valid_generator = data_collector(args.trainset, args.validset, args.useDA,args.subset, 0.99, params)
    seed = 42
    step=2
    gm=0.95
    torch.manual_seed(seed)

    num_devices = min(args.ngpus, torch.cuda.device_count())
    print(f"Testing {num_devices} GPUs.")
    pl_module = ColorVoxelDNNTrainModule(args.nores, args.dim, args.lr, step, gm)
    # for finetuning from a pretrained model
    #pl_module = ColorVoxelDNNTrainModule.load_from_checkpoint(args.input_model_path, strict=False,no_res=args.nores,dim=args.dim, lr=args.lr, lrstep=step, gm=gm)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath= args.saving_model_path + args.flag +'/',
        filename=  "best_val_checkpoint_model_"+"_lr_"+str(args.lr)+"_b_"+ str(args.batch)+"_da_"+ str(args.useDA)+"_nores_"+str(args.nores)+ "_nofil_"+str(args.nofilters)+"_nomix_"+str(args.nomix)+"_schedule_"+str(step)+str(gm)+"_dim_"+str(args.dim)+"-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
        #save_weights_only=True,
    )
    train_checkpoint_callback = ModelCheckpoint(
        monitor="loss",
        dirpath=args.saving_model_path + args.flag + '/',
        filename="best_trainloss_ckpt_" + "_lr_" + str(args.lr) + "_b_" + str(args.batch) + "_da_" + str(
            args.useDA) + "_nores_" + str(args.nores) + "_nofil_" + str(args.nofilters) + "_nomix_" + str(
            args.nomix) + "_schedule_" + str(step) + str(gm) + "_dim_" + str(args.dim) + "-{epoch:02d}-{val_loss:.2f}",
        save_top_k=2,
        mode="min",
        # save_weights_only=True,
    )

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=20, verbose=False, mode="min")
    tb_logger = pl_loggers.TensorBoardLogger(args.saving_model_path + args.flag + '/log/')
    trainer = Trainer(auto_lr_find=True,max_epochs=-1, gpus=num_devices, strategy="ddp",callbacks=[checkpoint_callback, early_stop_callback,train_checkpoint_callback],logger=tb_logger,accumulate_grad_batches=args.bacc)

    trainer.fit(pl_module, training_generator, valid_generator)


class ColorVoxelDNNTrainModule(LightningModule):
    def __init__(self,no_res,dim, lr, lrstep, gm):
        super().__init__()
        self.model=ResidualColorVoxelDNN(dim+1,  no_res,dim )
        self.dimension=dim
        self.lossfc=nn.CrossEntropyLoss()
        self.step=lrstep
        self.gm=gm
        self.lr=lr * 1e-5
        self.train_loss=0.
        self.valid_loss=0.
        self.valid_loss_min=np.Inf

    def forward(self, x,y):
        return self.model(x,y)


    def training_step(self, batch, batch_idx):
        coords, feats, occups = batch
        prev_feats = feats[:, :self.dimension]
        curr_feat = feats[:,:(self.dimension+1)] #include the last feature also, as it is already available
        sparse_inputs = ME.SparseTensor(curr_feat, coords)
        sparse_prev_inputs = ME.SparseTensor(prev_feats, coords)

        predicts=self.model(sparse_inputs,sparse_prev_inputs)
        if (self.dimension != 0):
            true_input = torch.round((feats[:, int(self.dimension)] * 255.5 + 255.5))
        else:
            true_input = torch.round((feats[:, int(self.dimension)] * 255.5 + 255.5 - 127.5))
        loss=loss_op(self.lossfc,true_input, predicts)


        if(batch_idx%200==0):
            Top1_accuracy, Top3_accuracy, Top5_accuracy, Top20_accuracy, fmarco, fmicro, fweight = compute_metrics(predicts, true_input)
            self.log('T1', Top1_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('T3', Top3_accuracy, on_step=True, on_epoch=True, prog_bar=False, logger=True)
            self.log('T5', Top5_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("loss", loss,on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.train_loss =  self.train_loss + ((1 / (batch_idx + 1)) * (loss.item() - self.train_loss))
        return loss

    def validation_step(self, batch, batch_idx):
        coords, feats, occups = batch
        prev_feats = feats[:, :self.dimension]
        curr_feat = feats[:, :(self.dimension+1)]
        sparse_inputs = ME.SparseTensor(curr_feat, coords)
        sparse_prev_inputs = ME.SparseTensor(prev_feats, coords)

        predicts = self.model(sparse_inputs, sparse_prev_inputs)
        if (self.dimension != 0):
            true_input = torch.round((feats[:, int(self.dimension)] * 255.5 + 255.5))
        else:
            true_input = torch.round((feats[:, int(self.dimension)] * 255.5 + 255.5 - 127.5))
        loss=loss_op(self.lossfc, true_input, predicts)
        if (batch_idx % 100 == 0):
            Top1_accuracy, Top3_accuracy, Top5_accuracy, Top20_accuracy, fmarco, fmicro, fweight = compute_metrics(
                predicts, true_input)
            self.log('vT1', Top1_accuracy, on_step=True, on_epoch=True, prog_bar=False, logger=True)
            self.log('vT3', Top3_accuracy, on_step=True, on_epoch=True, prog_bar=False, logger=True)
            self.log('vT5', Top5_accuracy, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.valid_loss = self.valid_loss + ((1 / (batch_idx + 1)) * (loss.item() - self.valid_loss))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr , betas=(0.9, 0.999))
        scheduler = lr_scheduler.StepLR(optimizer, step_size=self.step, gamma=self.gm)
        return [optimizer], [scheduler]

    def validation_epoch_end(self,output):
        self.log("val_loss",self.valid_loss,on_epoch=True, prog_bar=False, logger=True)
        if (self.valid_loss <= self.valid_loss_min):
            self.valid_loss_min = self.valid_loss

        self.valid_loss=0.
        self.train_loss = 0.
    def training_epoch_end(self,output):

        self.log("train_loss", self.train_loss,on_epoch=True, prog_bar=False, logger=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='color voxeldnn training')
    parser.add_argument("-blocksize", type=int, default=64, help="input block size")
    parser.add_argument("-batch", type=int, default=1, help="batch size")
    parser.add_argument("-opt", type=int, default=1, help="Optimizer selection")
    parser.add_argument("-nomix", type=int, default=10, help="number of mixture")
    parser.add_argument("-nores", type=int, default=2, help="number of mixture")
    parser.add_argument("-nofilters", type=int, default=256, help="number of mixture")
    parser.add_argument("-lr", type=int, default=10, help="actual lr=lr*1e-4")
    parser.add_argument("-subset", type=float, default=1.0, help="subset portion for mock training")

    parser.add_argument("-useDA", '--useDA', type=int,
                        default=3,
                        help='0: no data augmentation, 1: only rotation, 2: only subsampling, 3: both, 8: RGB to YCoCg and then subsampling')
    parser.add_argument("-dim", '--dim', type=int,
                        default=0,
                        help='0: red, 1: green, 2:blue')
    parser.add_argument("-color", '--color', type=bool,
                        default=True, action=argparse.BooleanOptionalAction,
                        help='color training or occupancy training')
    parser.add_argument("-usecuda", '--usecuda', type=bool,
                        default=True, action=argparse.BooleanOptionalAction,
                        help='using data augmentation or not')
    parser.add_argument("-flag", '--flag', type=str, default='test', help='training flag ')
    parser.add_argument("-trainset", '--trainset', action='append', type=str, help='path to train set ')
    parser.add_argument("-validset", '--validset', action='append', type=str, help='path to valid set ')
    parser.add_argument("-outputmodel", '--saving_model_path', type=str, help='path to output model file')
    parser.add_argument("-inputmodel", '--input_model_path', type=str, help='path to input model file')
    parser.add_argument("-ngpus", type=int, default=1, help="num_gpus")
    parser.add_argument("-bacc", type=int, default=1, help="gradient accumulation4")
    args = parser.parse_args()


    print("Starting Color training....")
    train_MoL_ColorVoxelDNN(args)


# python3 -m Training.cnet_attribute_training -trainset ../Datasets/CNeT_TrainingSet/33K/  -validset ../Datasets/CNeT_TrainingSet/33K/ -flag test -outputmodel Model/  -lr 7 -useDA 8   --color -opt 1 -dim 2 -ngpus 1  -batch 2  -bacc 1

