import torch
import torchvision.utils

import initializations as ini
import config as c

# Our training will be perfrom in this seperate file

for epochs in range(c.epochs):
    for batch_idx, (real, _) in enumerate(ini.train_loader):
        real = real.view(-1, 784).to(c.device) # reshape real images
        batch_size = real.shape[0]

        #### Setup training for Discriminator #### max log (D(real)) + log(1 - D(G(z)))

        noise = torch.rand(batch_size, c.z).to(c.device)
        fake = ini.g(noise) # first part of log function is implemented here!
        dis_real = ini.d(real).view(-1)
        lossD_real = ini.criterion(dis_real, torch.ones_like(dis_real))
        dis_fake = ini.d(fake).view(-1)
        lossD_fake = ini.criterion(dis_fake, torch.zeros_like(dis_real)) # we are sending zeros adn second part of the function
        lossD = (lossD_fake + lossD_real) / 2

        # optmization

        ini.g.zero_grad()
        lossD.backward(retain_graph=True)
        ini.optimizerd.step()

        # We want to now train Generator and want to min log(1 - D(G(z))) <--> log(max D(g(z)))

        output = ini.d(fake).view(-1)
        lossG = ini.criterion(output, torch.ones_like(output))

        ini.g.zero_grad()
        lossG.backward()
        ini.optimizerg.step()

        if batch_idx == 0:
            print(
                f" Epochs [{epochs}/{c.epochs}] \ "
                f"LossD: {lossD:.4f}, LossG: {lossG: .4f}"
            )

            with torch.no_grad():
                fake = ini.g(ini.fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                ini.writer_fake.add_image(
                    "MNIST Fake Images", img_grid_fake, global_step=ini.step
                )

                ini.writer_real.add_image(
                    "MNIST Real Images", img_grid_real, global_step=ini.step
                )

                ini.step += 1









