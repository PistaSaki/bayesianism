import keras
import keras.layers as kl
import keras.backend as K

class KLDivergenceLayer(kl.Layer):

    """ Identity transform layer that adds KL divergence
    to the final model loss.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):

        mu, sigma = inputs
        log_var = K.log(sigma**2)

        kl_batch = - .5 * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)

        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs


def get_stochastic_encoder(inferer, name = "encoder"):
    x = inferer.input
    z_mu, z_sigma = inferer.output
    assert len(z_mu.shape) == len(z_sigma.shape) == 2
    latent_dim = z_mu.shape[-1]
    
    ## Auxiliary layer adding Kullback-Leibler part of the loss
    kl_loss_layer = KLDivergenceLayer(name = "kl_loss_layer")
    z_mu, z_sigma = kl_loss_layer([z_mu, z_sigma])
    
    ## New random "input"
    eps = kl.Input(tensor=K.random_normal(shape=(K.shape(x)[0], latent_dim)), name ='eps')
    
    ## reparametrization
    z_eps = kl.Multiply(name='z_eps')([z_sigma, eps])
    z = kl.Add(name='z')([z_mu, z_eps])
    
    ##
    encoder = keras.Model(
        inputs = [x, eps],
        outputs = z,
        name = name
    )
    
    return encoder

def get_vae(inferer, decoder, name = "vae"):
    encoder = get_stochastic_encoder(inferer)
    x, eps = encoder.input
    z = encoder.output
    x_pred = decoder(z)
    
    return keras.Model(inputs=[x, eps], outputs=x_pred, name = name)
    
    