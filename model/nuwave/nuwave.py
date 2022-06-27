#Some codes are adopted from
#https://github.com/ivanvovk/WaveGrad
#https://github.com/lmnt-com/diffwave
#https://github.com/lucidrains/denoising-diffusion-pytorch
#https://github.com/hojonathanho/diffusion
import torch
import torch.nn as nn

from model.nuwave.model import NuWave as model

@torch.jit.script
def lognorm(pred, target):
    return (pred - target).abs().mean(dim=-1).clamp(min=1e-20).log().mean()

class NuWave(nn.Module):
    def __init__(self, train=True):
        super().__init__()
        self.device="cuda"
        self.hparams = {}
        self.hparams['arch'] = {
            "residual_layers": 30,
            "residual_channels": 64,
            "dilation_cycle_length": 10,
            "pos_emb_dim": 512
        }
        self.hparams['ddpm'] = {
            'max_step': 1000,
            'noise_schedule': "torch.linspace(1e-6, 0.006, 1000)",
            'pos_emb_scale': 50000,
            'pos_emb_channels': 128 ,
            'infer_step': 8,
            'infer_schedule': "torch.tensor([1e-6,2e-6,1e-5,1e-4,1e-3,1e-2,1e-1,9e-1])",
            "pos_emb_scale": 50000,
            "pos_emb_channels": 128
        }
        # self.hparams['audio'] = {
        #     "sr": 48000,
        #     "nfft": 1024,
        #     "hop": 256,
        #     "ratio": 2,
        # }

        self.model = model(self.hparams).to(self.device)
        self.norm = nn.L1Loss()  #loss
        self.set_noise_schedule(self.hparams, train)
    
    # DDPM backbone is adopted form https://github.com/ivanvovk/WaveGrad
    def set_noise_schedule(self, hparams, train=True):
        self.max_step = self.hparams['ddpm']['max_step'] if train \
                else self.hparams['ddpm']['infer_step']
        noise_schedule = eval(self.hparams['ddpm']['noise_schedule']) if train \
                else eval(self.hparams['ddpm']['infer_schedule'])

        self.register_buffer('betas', noise_schedule, False)
        self.register_buffer('alphas', 1 - self.betas, False)
        self.register_buffer('alphas_cumprod', self.alphas.cumprod(dim=0),
                             False)
        self.register_buffer(
            'alphas_cumprod_prev',
            torch.cat([torch.FloatTensor([1.]), self.alphas_cumprod[:-1]]),
            False)
        alphas_cumprod_prev_with_last = torch.cat(
            [torch.FloatTensor([1.]), self.alphas_cumprod])
        self.register_buffer('sqrt_alphas_cumprod_prev',
                             alphas_cumprod_prev_with_last.sqrt(), False)
        self.register_buffer('sqrt_alphas_cumprod', self.alphas_cumprod.sqrt(),
                             False)
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             (1. / self.alphas_cumprod).sqrt(), False)
        self.register_buffer('sqrt_alphas_cumprod_m1',
                             (1. - self.alphas_cumprod).sqrt() *
                             self.sqrt_recip_alphas_cumprod, False)
        posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) \
                             / (1 - self.alphas_cumprod)
        posterior_variance = torch.stack(
                            [posterior_variance,
                             torch.FloatTensor([1e-20] * self.max_step)])
        posterior_log_variance_clipped = posterior_variance.max(
            dim=0).values.log()
        posterior_mean_coef1 = self.betas * self.alphas_cumprod_prev.sqrt() / (
            1 - self.alphas_cumprod)
        posterior_mean_coef2 = (1 - self.alphas_cumprod_prev
                                ) * self.alphas.sqrt() / (1 -
                                                          self.alphas_cumprod)
        self.register_buffer('posterior_log_variance_clipped',
                             posterior_log_variance_clipped, False)
        self.register_buffer('posterior_mean_coef1',
                             posterior_mean_coef1, False)
        self.register_buffer('posterior_mean_coef2',
                             posterior_mean_coef2, False)

    def sample_continuous_noise_level(self, step):
        rand = torch.rand_like(step, dtype=torch.float, device=step.device)
        continuous_sqrt_alpha_cumprod = \
                self.sqrt_alphas_cumprod_prev[step - 1] * rand \
                + self.sqrt_alphas_cumprod_prev[step] * (1. - rand)
        # return continuous_sqrt_alpha_cumprod.unsqueeze(-1)
        return continuous_sqrt_alpha_cumprod[...,None, None]

    def q_sample(self, y_0, step=None, noise_level=None, eps=None):
        batch_size = y_0.shape[0]
        if noise_level is not None:
            continuous_sqrt_alpha_cumprod = noise_level
        elif step is not None:
            continuous_sqrt_alpha_cumprod = self.sqrt_alphas_cumprod_prev[step]
        assert (step is not None or noise_level is not None)
        if isinstance(eps, type(None)):
            eps = torch.randn_like(y_0, device=y_0.device)
        outputs = continuous_sqrt_alpha_cumprod * y_0 + (
            1. - continuous_sqrt_alpha_cumprod**2).sqrt() * eps
        return outputs

    def q_posterior(self, y_0, y, step):
        posterior_mean = self.posterior_mean_coef1[step] * y_0  \
                         + self.posterior_mean_coef2[step] * y
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[step]
        return posterior_mean, posterior_log_variance_clipped

    @torch.no_grad()
    def predict_start_from_noise(self, y, t, eps):
        return self.sqrt_recip_alphas_cumprod[t].unsqueeze(
            -1) * y - self.sqrt_alphas_cumprod_m1[t].unsqueeze(-1) * eps

    # t: interger not tensor
    @torch.no_grad()
    def p_mean_variance(self, y, y_down, t, clip_denoised: bool):
        batch_size = y.shape[0]
        noise_level = self.sqrt_alphas_cumprod_prev[t + 1].repeat(
            batch_size, 1)
        eps_recon = self.model(y, y_down, noise_level)
        y_recon = self.predict_start_from_noise(y, t, eps_recon)
        if clip_denoised:
            y_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_log_variance_clipped = self.q_posterior(
            y_recon, y, t)
        return model_mean, posterior_log_variance_clipped

    @torch.no_grad()
    def compute_inverse_dynamincs(self, y, y_down, t, clip_denoised=False):
        # Calculate the normal distribution at this time step.
        model_mean, model_log_variance = self.p_mean_variance(
            y, y_down, t, clip_denoised)
        eps = torch.randn_like(y) if t > 0 else torch.zeros_like(y)
        return model_mean + eps * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def sample(self, y_down,
               start_step=None,
               init_noise=True,
               store_intermediate_states=False):
        batch_size = y_down.shape[0]
        start_step = self.max_step if start_step is None \
                else min(start_step, self.max_step)
        step = torch.tensor([start_step] * batch_size,
                            dtype=torch.long,
                            device=self.device)
        bs, length = y_down.shape
        expect_shape = y_down.unsqueeze(1).expand(bs, 2, length).shape
        y_t = torch.randn(
                expect_shape, device=self.device) if init_noise \
                else self.q_sample(y_down, step=step)
        ys = [y_t]
        t = start_step - 1
        while t >= 0:
            y_t = self.compute_inverse_dynamincs(y_t, y_down, t)
            ys.append(y_t)
            t -= 1
        return ys if store_intermediate_states else ys[-1]

    def forward(self, x, x_clean, noise_level):
        x = self.model(x, x_clean, noise_level)
        return x

    def common_step(self, y, y_low, step):
        noise_level = self.sample_continuous_noise_level(step) \
                if self.training \
                else self.sqrt_alphas_cumprod_prev[step].unsqueeze(-1)
        eps = torch.randn_like(y, device=y.device)
        y_noisy = self.q_sample(y, noise_level=noise_level, eps=eps)

        eps_recon = self.model(y_noisy, y_low, noise_level.squeeze(-1))
        loss = lognorm(eps_recon, eps)
        return loss, y, y_low, y_noisy, eps, eps_recon

    def training_step(self, batch, batch_nb=None):
        wav, wav_l = batch
        step = torch.randint(
            0, self.max_step, (wav.shape[0], ), device=self.device) + 1
        loss, *_ = self.common_step(wav, wav_l, step)
        return loss

    def test_step(self, wav_l, batch_nb=None):
        wav_up = self.sample(wav_l, self.hparams['ddpm']['infer_step'])
        return wav_up

if __name__ == "__main__":
    model = NuWave().cuda()
    # pitch, energy prediction, embedding g
    wav, wav_l = torch.randn((3,2,100)).cuda(), torch.randn((3,100)).cuda()
    loss = model.training_step((wav, wav_l))
    print(loss)
    wav_up = model.test_step(wav_l=wav_l)
    print(wav_up.size())