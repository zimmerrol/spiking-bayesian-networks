import numpy as np
# import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import animation
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
import utility as ut
import network as nt
from tqdm import tqdm as tqdm
import plot as pt
from collections import deque # fifo structure for history
import threading



image_duration = 0.250    # [s]
mnist_px_samples = None
current_image_px = None
current_image_id = None
current_image_t = -1      # [units of image_duration]
current_time = 0.0        # [s]

# mnist
labels = [0, 1]
class mnist():
    def __init__(self):
        (x_train, y_train), (x_test, y_test) = ut.mnist.load_data()
        self.selection = [y_test == label for label in labels]

        minimum_length = min(np.sum(self.selection, axis=1))
        self.selection = np.any([np.all((item, np.cumsum(item) < minimum_length), axis=0) for item in self.selection], axis=0)
        self.X = x_test[self.selection]
        self.Y = y_test[self.selection]
        self.X = self.X.reshape((len(self.X), -1)) / 255.0

        global mnist_px_samples
        mnist_px_samples = (self.X > 0.5).astype(np.float32)


def get_input_at_time(t):
    """
        which input image is shown at time t
        updates current image, only works for consecutive t

        Parameters
        ----------
        t : float
            time in seconds
    """

    global current_image_t
    global current_image_id
    global current_image_px

    assert t >= current_image_t * image_duration, "don't travel back in time"

    if t / image_duration > current_image_t + 1:
        current_image_t = np.floor(t / image_duration)
        current_image_id = np.random.randint(0, len(mnist_px_samples))
        current_image_px = mnist_px_samples[current_image_id]

    return current_image_px


class EventBinaryWTANetwork():
    """

    Eventbased mplementation of the (soft) Winner Takes All (WTA) network,
    assuming Bernoulli distributions
    """

    def __init__(self, n_inputs, n_outputs, delta_T, r_net, m_k, eta_v, eta_b,
        history_duration = 0):
        """
            Parameters
            ----------
            delta_T : float
                numeric integration time step size

            r_net : float
                global target firing rate of the whole network [1/s]

            m_k :  numpy.ndarray
                size n_outputs
                has to sum up to one, propto relative target firing rate of
                neuron k

            eta_v : float
                learning rate of v eq (8)

            eta_b : float
                learing rate of b eq. (7)

            history_duration : float
                how long [s] to store input and output history
        """

        self._n_inputs = n_inputs
        self._n_outputs = n_outputs
        self._delta_T = delta_T
        self._r_net = r_net
        self._m_k = m_k
        self._eta_v = eta_v
        self._eta_b = eta_b
        self._history_duration = history_duration
        self._l_avg = 0              # current avg likelihood estimate over past history duration
        self._l_hist = deque([])     # log likelihoods at past time steps
        self._z_hist = deque([])     # index k of spiking output neuron z
        self._t_hist = deque([])     # time float of spike
        self._y_hist = deque([])     # mnist sample id of input

        # for likelihood plotting keep integer time step data [s]
        self._l_avg_hist = deque([])
        self._T_hist = deque([])
        self._T_duration = 100 # [s]

        if np.isscalar(self._m_k):
            self._m_k = np.ones((n_outputs, 1)) * self._m_k

        assert len(self._m_k) == n_outputs, "Length of m_ks does not match number of output neurons"

        self._V = np.random.normal(scale=1e-3, size=(self._n_outputs, self._n_inputs))
        self._b = np.zeros((self._n_outputs, 1))
        # output spike count to get spikes per second
        self._z_spikes = np.zeros(self._n_outputs)

    def next_spike_time(self):
        """
            Return next spike time from poisson process with rate r_net
        """
        return - np.log(np.random.uniform())/self._r_net

    def init_z_plot(self):
        assert self._history_duration > 0, "need a history to plot"
        self._z_fig, self._z_ax = plt.subplots()
        ax = plt.axes(xlim=(-self._history_duration, 0),
            ylim=(-.5, net._n_outputs-.5))
        self._z_scat = ax.scatter([], [], s=10)
        # self._z_fig.show()

    def update_z_plot(self):
        dat = np.array(list(zip(net._t_hist, net._z_hist)))
        dat[:,0] = dat[:,0] - current_time
        self._z_scat.set_offsets(dat)
        # self._z_fig.draw()

    def init_y_plot(self):
        assert self._history_duration > 0, "need a history to plot"
        self._y_fig, self._y_ax = plt.subplots()
        ax = plt.axes(xlim=(-self._history_duration, 0),
            ylim=(-.5, net._n_inputs-.5))
        self._y_scat = ax.scatter([], [], s=10)
        # plt.show(block=False)


    def init_weight_plot(self):
        self._w_fig = plt.figure(figsize=(3.5, 1.16), dpi=300)
        axes = pt.add_axes_as_grid(self._w_fig, 2, 6, m_xc=0.01, m_yc=0.01)
        self._w_imshows = []
        for i, ax in enumerate( list(axes.flatten()) ):
            # disable legends
            ax.set_yticks([])
            ax.set_xticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)

            self._w_imshows.append(ax.imshow(ut.sigmoid(
                net._V[i].reshape((28, 28))), vmin=0.3, vmax=.7))


    def update_weight_plot(self):
        weights = self._V.reshape((-1, 28, 28))
        for i in range(len(self._w_imshows)):
            self._w_imshows[i].set_data(ut.sigmoid(weights[i]))


    def init_pca_plot(self, mnist):
        # set up figure for PCA
            self._pca_fig, self._pca_ax = plt.subplots(1)
            colors = ["C0", "C1", "C2", "C3"]
            Y_color = np.empty(mnist.Y.shape, dtype="object")
            for i, label in enumerate(labels):
                Y_color[mnist.Y == label] = colors[i%len(colors)]
            self._pca = PCA(n_components=2)
            self._pca.fit(mnist.X)
            X_pca = self._pca.transform(mnist.X)

            self._pca_ax.set_xlabel("PC 1")
            self._pca_ax.set_ylabel("PC 2")

            self._pca_scatter_constant = self._pca_ax.scatter(
                X_pca[:, 0], X_pca[:, 1], c=Y_color,alpha=0.5, marker="o", s=2)
            self._pca_scatter_variable = self._pca_ax.scatter(
                np.zeros(self._n_outputs), np.zeros(self._n_outputs),
                c="black", marker="o", s=30)

            self._pca_fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0),
                fancybox=True, ncol=1+len(labels),
                handles=[mpatches.Patch(
                    color=Y_color[i], label=labels[i]) \
                    for i in range(len(labels))] + \
                [mpatches.Patch(color="black", label="Weights")])


    def update_pca_plot(self):
        weights_pca = self._pca.transform(ut.sigmoid(self._V))
        self._pca_scatter_variable.set_offsets(weights_pca)


    def init_l_plot(self):
        assert self._history_duration > 0, "need a history to plot"
        self._l_fig, _ = plt.subplots()
        self._l_ax = plt.axes(xlim=(0, self._history_duration),
            ylim=(-360, -150))
        self._l_ax.set_xlabel(r'time $[s]$')
        self._l_ax.set_ylabel(r'$<L(y)>$')
        self._l_line, = self._l_ax.plot(self._T_hist, self._l_avg_hist)

    def update_l_plot(self):
        xmin, xmax = self._l_ax.get_xlim()
        ymin, ymax = self._l_ax.get_ylim()
        # x axis rescale if needed
        if self._T_hist[-1] > xmax:
            # set both
            # if (xmax-xmin) > self._T_duration:
            #     self._l_ax.set_xlim(
            #         self._T_hist[-1]-self._T_duration, self._T_hist[-1])
            # else:
            self._l_ax.set_xlim(xmin, self._T_hist[-1])
            self._l_ax.figure.canvas.draw()
        if self._l_avg_hist[-1] > ymax:
            ymax = self._l_avg_hist[-1] + np.abs(self._l_avg_hist[-1] * .1)
            self._l_ax.set_ylim(ymin, ymax)

        self._l_line.set_data(self._T_hist, self._l_avg_hist)

    def step(self):
        global current_time
        isi =  - np.log(np.random.uniform())/self._r_net
        inputs = get_input_at_time(current_time + isi)
        inputs = inputs.reshape((-1, 1))
        assert len(inputs) == self._n_inputs, "Input length does not match"

        # u = V * input + b
        u = np.dot(self._V, inputs) + self._b

        z = np.zeros((self._n_outputs, 1))

        # p \propto softmax(u); eq. (4)
        # p_z = np.exp(u) / np.sum(np.exp(u) + 1e-8) * self._delta_T * self._r_net
        p_z = np.exp(u) / np.sum(np.exp(u) + 1e-8)

        # erm so this is needed cos event based i guess
        # p_z = p_z / np.sum(p_z)


        # sample from softmax distribution, i.e. choose a single neuron to spike
        sum_p_z = np.cumsum(p_z)
        diff = sum_p_z - np.random.uniform(0, 1, 1) > 0
        k = np.argmax(diff)

        # print(p_z, np.sum(p_z), sum_p_z)
        if diff[k]:
            z[k] = 1.0
            self._z_spikes[k] += 1

        self._b += self._eta_b * \
            (isi * self._r_net * self._m_k - ut.dirac(z - 1))
        self._V += self._eta_v * ut.dirac(z - 1) * \
            (inputs.T - ut.sigmoid(self._V))

        current_time = current_time + isi

        # update history stuff
        if self._history_duration > 0:
            self._t_hist.append(current_time)
            self._z_hist.append(k)
            self._y_hist.append(current_image_id)
            # log likelihood
            try:
                Ak = np.sum(np.log(1+np.exp(self._V)), -1)
                pi = ut.sigmoid(self._V)
                l = np.log(1.0/self._n_outputs) + \
                    np.log(np.sum(np.prod(inputs[:,0] * pi + \
                    (1-inputs[:,0]) * (1-pi), axis=-1)))
            except:
                l = np.nan
            self._l_hist.append(l)

            while len(self._t_hist) > 1 \
            and self._t_hist[0] < current_time - self._history_duration:
                self._t_hist.popleft()
                self._z_hist.popleft()
                self._y_hist.popleft()
                self._l_hist.popleft()

            self._l_avg = np.nanmean(self._l_hist)

            # update hidden long history
            if len(self._T_hist) == 0 \
            or np.floor(current_time) > self._T_hist[-1]:
                self._T_hist.append(np.floor(current_time))
                self._l_avg_hist.append(self._l_avg)

            # if len(self._T_hist) > self._T_duration:
            #     self._T_hist.popleft()
            #     self._l_avg_hist.popleft()

        return z

    def update_steps(self, steps=10000):

        pbar = tqdm(range(steps))
        for i in pbar:
            z = self.step()[:,0]

            # update weight plot every percent
            if not i % int(steps/100):
                # self.update_weight_plot()

                out = ''
                for i in net._z_spikes:
                    out += f'{i/current_time:.1f} '
                out += f'<L(y)> = {self._l_avg:.1f}'
                pbar.set_description(out)

            # self.update_z_plot()

    def init_animations(self):
        # animate spike train
        def anim_z(i):
            try:
                self.update_z_plot()
            except:
                pass
        self._z_animation = animation.FuncAnimation(self._z_fig, anim_z,
            init_func=None, frames=2, interval=200,
            blit=False, repeat=True)

        # animate weights
        def anim_w(i):
            try:
                self.update_weight_plot()
            except:
                pass
        self._w_animation = animation.FuncAnimation(self._w_fig, anim_w,
            init_func=None, frames=2, interval=1500,
            blit=False, repeat=True)

        def anim_pca(i):
            try:
                self.update_pca_plot()
            except:
                pass
        self._pca_animation = animation.FuncAnimation(self._pca_fig, anim_pca,
            init_func=None, frames=2, interval=1500,
            blit=False, repeat=True)

        def anim_l(i):
            try:
                self.update_l_plot()
            except:
                pass
        self._l_animation = animation.FuncAnimation(self._l_fig, anim_l,
            init_func=None, frames=2, interval=1500,
            blit=False, repeat=True)


def run_thread():
    thread = threading.Thread(target=net.update_steps)
    thread.daemon = True
    thread.start()
    return thread

delta_T = 1e-3
n_outputs = 12
n_inputs = 28*28
r_net = 50.0 # 0.5
m_k = 1.0/n_outputs

if __name__ == '__main__':
    mnist = mnist()
    get_input_at_time(0)
    net = EventBinaryWTANetwork(n_inputs=n_inputs, n_outputs=n_outputs,
        delta_T=delta_T, r_net=r_net, m_k=m_k, eta_v=1e-2, eta_b=1e-0,
        history_duration=10)

    # plt.ioff()
    plt.ion()
    net.init_weight_plot()
    net.init_z_plot()
    net.init_pca_plot(mnist)
    net.init_l_plot()
    net.init_animations()
    # plt.show() # dont do this, causes crashes on osx, use plt.ion() instead

    thread = run_thread()


