import torch 
import lightning as pl
from mlcolvar.cvs import BaseCV
from mlcolvar.core import FeedForward, Normalization
from mlcolvar.core.loss.kl_tda_loss import KL_TDA_loss

__all__ = ["KL_DeepTDA_CV"]

class KL_DeepTDA_CV(BaseCV, pl.LightningModule):
    """
    Define Deep Targeted Discriminant Analysis (Deep-TDA) CV.
    Combine the inputs with a neural-network and optimize it in a way such that the data are distributed accordingly to a target distribution.
    """

    BLOCKS = ['norm_in', 'nn']

    # TODO n_states optional?
    def __init__(self,
                n_states : int,
                n_cvs : int,
                target_centers : list, 
                target_sigmas : list, 
                layers : list,
                KLD_reg : str = None, 
                options : dict = {}, 
                **kwargs):
        """
        Define Deep Targeted Discriminant Analysis (Deep-TDA) CV.

        Parameters
        ----------
        n_states : int
            Number of states for the training
        n_cvs : int
            Numnber of collective variables to be trained
        target_centers : list
            Centers of the Gaussian targets
        target_sigmas : list
            Standard deviations of the Gaussian targets
        layers : list
            Number of neurons per layer
        KLD_reg : bool,
            Add a Kullback-Leibler divergence term to the loss to actually enforce the gaussian distribution, by default False.
            If True, use more epochs for training as it slows down the learning procedure
        options : dict[str, Any], optional
            Options for the building blocks of the model, by default {}.
            Available blocks: ['normIn', 'nn'].
            Set 'block_name' = None or False to turn off that block
        """

        super().__init__(in_features=layers[0], out_features=layers[-1], **kwargs)
        
        # =======   LOSS  =======
        self.loss_fn = KL_TDA_loss(n_states=n_states,
                                    target_centers=target_centers,
                                    target_sigmas=target_sigmas,
        )

        # ======= OPTIONS ======= 
        # parse and sanitize
        options = self.parse_options(options)
        # Save n_states        
        self.n_states = n_states
        if self.out_features != n_cvs:
            raise ValueError("Number of neurons of last layer should match the number of CVs!")
        
        # check size  and type of targets
        if not isinstance(target_centers,torch.Tensor):
            target_centers = torch.Tensor(target_centers)
        if not isinstance(target_sigmas,torch.Tensor):
            target_sigmas = torch.Tensor(target_sigmas)

        if target_centers.shape != target_sigmas.shape:
            raise ValueError(f"Size of target_centers and target_sigmas should be the same!")
        if n_states != target_centers.shape[0]:
            raise ValueError(f"Size of target_centers at dimension 0 should match the number of states! Expected {n_states} found {target_centers.shape[0]}")
        if len(target_centers.shape) == 2:
            if n_cvs != target_centers.shape[1]:
                raise ValueError((f"Size of target_centers at dimension 1 should match the number of cvs! Expected {n_cvs} found {target_centers.shape[1]}"))

        # Initialize normIn
        o = 'norm_in'
        if ( options[o] is not False ) and (options[o] is not None):
            self.norm_in = Normalization(self.in_features,**options[o])

        # initialize NN
        o = 'nn'
        self.nn = FeedForward(layers, **options[o])

    def training_step(self, train_batch, batch_idx):
        # =================get data===================
        x = train_batch['data']
        labels = train_batch['labels']
        # =================forward====================
        z = self.forward_cv(x)
        # ===================loss=====================
        loss = self.loss_fn(z, labels)
        # ====================log=====================+
        name = 'train' if self.training else 'valid'
        self.log(f'{name}_loss', loss, on_epoch=True,prog_bar=True, )
        return loss

# TODO signature of tests?
import numpy as np
def test_deeptda_cv():
    from mlcolvar.data import DictDataset
    from mlcolvar.data import DictModule

    for states_and_cvs in [  [1, 1]]:
        # get the number of states and cvs for the test run
        n_states = states_and_cvs[0]
        n_cvs = states_and_cvs[1]
        
        in_features, out_features = 2, n_cvs 
        layers = [in_features, 4, 2, out_features]
        target_centers = np.random.randn(n_states, n_cvs)
        target_sigmas = np.random.randn(n_states, n_cvs)

        # test initialize via dictionary
        options= { 'nn' : { 'activation' : 'relu' } }

        model = KL_DeepTDA_CV(n_states = n_states, n_cvs = n_cvs, target_centers = target_centers, target_sigmas = target_sigmas, layers = layers, options=options)
        
        print('----------')
        print(model)

        # create dataset
        samples = 100
        X = torch.randn((samples * n_states, 2))

        # create labels
        y = torch.zeros(X.shape[0])
        for i in range(1, n_states):
            y[samples*i:] += 1
        
        dataset = DictDataset({'data': X, 'labels' : y})
        datamodule = DictModule(dataset,lengths=[0.75,0.2,0.05], batch_size=samples)        
        # train model
        trainer = pl.Trainer(accelerator='cpu', max_epochs=2, logger=None, enable_checkpointing=False)
        trainer.fit( model, datamodule )

        # trace model
        traced_model = model.to_torchscript(file_path=None, method='trace', example_inputs=X[0])
        model.eval()
        assert torch.allclose(model(X),traced_model(X))

if __name__ == "__main__":
    test_deeptda_cv()