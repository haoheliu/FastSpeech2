import torch
import torch.nn as nn


class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2Loss, self).__init__()
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()


    def generator_loss(self, disc_outputs):
        loss = 0
        gen_losses = []
        for dg in disc_outputs:
            dg = dg.float()
            l = self.hhl_loss(torch.mean((1-dg)**2))
            gen_losses.append(l)
            loss += l
        return loss, gen_losses
    
    def feature_loss(self, fmap_r, fmap_g):
        loss = 0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                rl = rl.float().detach()
                gl = gl.float()
                loss += torch.mean(torch.abs(rl - gl))
        return loss * 2 

    def hhl_loss(self, x):
        return (-(1/(x-1+1e-8)))-1

    def discriminator_loss(self, disc_real_outputs, disc_generated_outputs):
        loss = 0
        r_losses = []
        g_losses = []
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            dr = dr.float()
            dg = dg.float()
            r_loss = self.hhl_loss(torch.mean((1-dr)**2)) 
            g_loss = self.hhl_loss(torch.mean(dg**2))
            loss += (r_loss + g_loss)
            r_losses.append(r_loss.item())
            g_losses.append(g_loss.item())
        return loss, r_losses, g_losses

    def forward(self, inputs, predictions):
        (
            mel_targets,
            _,
            _,
            pitch_targets,
            energy_targets,
            duration_targets,
        ) = inputs[6:]
        (
            mel_predictions,
            postnet_mel_predictions,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            _,
            src_masks,
            mel_masks,
            _,
            _,
        ) = predictions
        src_masks = ~src_masks
        mel_masks = ~mel_masks
        log_duration_targets = torch.log(duration_targets.float() + 1)
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False

        # if self.pitch_feature_level == "phoneme_level":
        #     pitch_predictions = pitch_predictions.masked_select(src_masks)
        #     pitch_targets = pitch_targets.masked_select(src_masks)
        # elif self.pitch_feature_level == "frame_level":
        #     pitch_predictions = pitch_predictions.masked_select(mel_masks)
        #     pitch_targets = pitch_targets.masked_select(mel_masks)

        # if self.energy_feature_level == "phoneme_level":
        #     energy_predictions = energy_predictions.masked_select(src_masks)
        #     energy_targets = energy_targets.masked_select(src_masks)
        # if self.energy_feature_level == "frame_level":
        #     energy_predictions = energy_predictions.masked_select(mel_masks)
        #     energy_targets = energy_targets.masked_select(mel_masks)

        # log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        # log_duration_targets = log_duration_targets.masked_select(src_masks)

        # mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        # postnet_mel_predictions = postnet_mel_predictions.masked_select(
        #     mel_masks.unsqueeze(-1)
        # )
        # mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))
        
        mel_loss = 45 * self.mae_loss(mel_predictions.masked_select(mel_masks.unsqueeze(-1)), mel_targets.masked_select(mel_masks.unsqueeze(-1)))
        postnet_mel_loss = 45 * self.mae_loss(postnet_mel_predictions.masked_select(mel_masks.unsqueeze(-1)), mel_targets.masked_select(mel_masks.unsqueeze(-1)))
        total_loss = mel_loss + postnet_mel_loss

        # if(torch.mean(pitch_predictions)>1e-5):
        #     pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        #     total_loss += pitch_loss
        # else:
        #     pitch_loss = torch.tensor([0.0]).to(pitch_predictions.device)
        
        # if(torch.mean(energy_predictions)>1e-5):
        #     energy_loss = self.mse_loss(energy_predictions, energy_targets)
        #     total_loss += energy_loss
        # else:
        #     energy_loss = torch.tensor([0.0]).to(energy_predictions.device)
        
        # if(torch.mean(log_duration_predictions)>1e-5):
        #     duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)
        #     total_loss += duration_loss
        # else:
        #     duration_loss = torch.tensor([0.0]).to(log_duration_predictions.device)
            
        total_loss = (
            total_loss
        )
        
        duration_loss = torch.tensor([0.0]).to(total_loss.device)
        pitch_loss = torch.tensor([0.0]).to(total_loss.device)
        energy_loss = torch.tensor([0.0]).to(total_loss.device)
        
        return (
            total_loss,
            mel_loss,
            postnet_mel_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
        ), (postnet_mel_predictions, mel_targets)
