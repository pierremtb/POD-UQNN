#%%
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import warnings
warnings.filterwarnings('ignore')
tfk = tf.keras
tfd = tfp.distributions
K = tfk.backend

#%%
def f(x, sigma):
    epsilon = np.random.randn(*x.shape) * sigma
    return x**3 + epsilon

train_size = 32
noise = 3.0

X = np.linspace(-4, 4, train_size).reshape(-1, 1)
y = f(X, sigma=noise)
y_true = f(X, sigma=0.0)

plt.scatter(X, y, marker='+', label='Training data')
plt.plot(X, y_true, label='Truth')
plt.title('Noisy training data and ground truth')
plt.legend()
plt.show()

tf.keras.backend.set_floatx('float32')

#%%
class DenseVariational(tfk.layers.Layer):
    def __init__(self,
                 units,
                 kl_weight,
                 activation=None,
                 prior_sigma_1=1.5,
                 prior_sigma_2=0.1,
                 prior_pi=0.5, **kwargs):
        self.units = units
        self.kl_weight = kl_weight
        self.activation = tfk.activations.get(activation)
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi_1 = prior_pi
        self.prior_pi_2 = 1.0 - prior_pi
        self.init_sigma = np.sqrt(self.prior_pi_1 * self.prior_sigma_1 ** 2 +
                                  self.prior_pi_2 * self.prior_sigma_2 ** 2)

        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units

    def build(self, input_shape):
        self.kernel_mu = self.add_weight(name='kernel_mu',
                                         shape=(input_shape[1], self.units),
                                         initializer=tfk.initializers.RandomNormal(stddev=self.init_sigma),
                                         trainable=True)
        self.bias_mu = self.add_weight(name='bias_mu',
                                       shape=(self.units,),
                                       initializer=tfk.initializers.RandomNormal(stddev=self.init_sigma),
                                       trainable=True)
        self.kernel_rho = self.add_weight(name='kernel_rho',
                                          shape=(input_shape[1], self.units),
                                          initializer=tfk.initializers.constant(0.0),
                                          trainable=True)
        self.bias_rho = self.add_weight(name='bias_rho',
                                        shape=(self.units,),
                                        initializer=tfk.initializers.constant(0.0),
                                        trainable=True)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        kernel_sigma = tf.math.softplus(self.kernel_rho)
        kernel = self.kernel_mu + kernel_sigma * tf.random.normal(self.kernel_mu.shape)

        bias_sigma = tf.math.softplus(self.bias_rho)
        bias = self.bias_mu + bias_sigma * tf.random.normal(self.bias_mu.shape)

        self.add_loss(self.kl_loss(kernel, self.kernel_mu, kernel_sigma) +
                      self.kl_loss(bias, self.bias_mu, bias_sigma))

        return self.activation(K.dot(inputs, kernel) + bias)

    def kl_loss(self, w, mu, sigma):
        variational_dist = tfp.distributions.Normal(mu, sigma)
        return self.kl_weight * K.sum(variational_dist.log_prob(w) - self.log_prior_prob(w))

    def log_prior_prob(self, w):
        comp_1_dist = tfp.distributions.Normal(0.0, self.prior_sigma_1)
        comp_2_dist = tfp.distributions.Normal(0.0, self.prior_sigma_2)
        return K.log(self.prior_pi_1 * comp_1_dist.prob(w) +
                     self.prior_pi_2 * comp_2_dist.prob(w))

batch_size = train_size
num_batches = train_size / batch_size

kl_weight = 1.0 / num_batches
prior_params = {
    'prior_sigma_1': 1.5, 
    'prior_sigma_2': 0.1, 
    'prior_pi': 0.5 
}

def split_mean_var(data):
    mean, out_var = tf.split(data, num_or_size_splits=2, axis=1)
    var = tf.math.log(1.0 + tf.exp(out_var)) + 1e-6
    return [mean, var]

x_in = tfk.Input(shape=(1,))
x = DenseVariational(20, kl_weight, **prior_params, activation='relu')(x_in)
x = DenseVariational(20, kl_weight, **prior_params, activation='relu')(x)
x = DenseVariational(1 + 1, kl_weight, **prior_params)(x)
# x = tf.keras.layers.Lambda(split_mean_var)(x)
x = tfp.layers.DistributionLambda(lambda t: tfd.Normal(
    loc=t[..., :1], scale=tf.math.log(1.0 + tf.exp(t[..., :1])) + 1e-6))(x)
model = tfk.Model(x_in, x)

def neg_log_likelihood(y_obs, y_pred):
    # dist = tfp.distributions.Normal(loc=y_pred[0], scale=y_pred[1])
    dist = y_pred
    return K.sum(-dist.log_prob(y_obs))

#%%

# X = tf.convert_to_tensor(X, dtype="float64")
# y = tf.convert_to_tensor(y, dtype="float64")
# optimizer = tfk.optimizers.Adam(lr=0.05)
# epochs=5000
# for e in range(epochs):
#   with tf.GradientTape() as tape:
#     loss_value = neg_log_likelihood(y, model(X))
#   grads = tape.gradient(loss_value, model.trainable_variables)
#   optimizer.apply_gradients(zip(grads, model.trainable_variables))
#   if e % 100 == 0:
#     print(f"#{e}: {loss_value:4e}")

model.compile(loss=neg_log_likelihood, optimizer=tfk.optimizers.Adam(lr=0.01))
#model.compile(loss=custom_loss(sigma), optimizer=tfk.optimizers.Adam(lr=0.001), experimental_run_tf_function=False)
model.fit(X, y, batch_size=batch_size, epochs=5000, verbose=2)

#%%
import tqdm

X_test = np.linspace(-6, 6, 1000).reshape(-1, 1)
y_pred_list = []

for i in tqdm.tqdm(range(500)):
    dist = model(X_test)
    y_pred, y_pred_sig = dist.mean().numpy(), 0
    y_pred_list.append(y_pred)
    
y_preds = np.concatenate(y_pred_list, axis=1)

y_mean = np.mean(y_preds, axis=1)
y_sigma = np.std(y_preds, axis=1)

plt.plot(X_test, y_mean, 'r-', label='Predictive mean');
plt.scatter(X, y, marker='+', label='Training data')
plt.fill_between(X_test.ravel(), 
                 y_mean + 2 * y_sigma, 
                 y_mean - 2 * y_sigma, 
                 alpha=0.5, label='Epistemic uncertainty')
plt.title('Prediction')
plt.legend()
plt.show()