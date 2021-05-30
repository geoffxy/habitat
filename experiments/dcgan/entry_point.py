import torch
import torch.nn as nn

from . import dcgan


def skyline_model_provider(numgpu=1):
    netG = dcgan.Generator(numgpu).cuda()
    netG.apply(dcgan.weights_init)
    netD = dcgan.Discriminator(numgpu).cuda()
    netD.apply(dcgan.weights_init)
    return netG, netD


def skyline_input_provider(batch_size=64):
    return (
        batch_size,
        torch.randn((batch_size, 3, 64, 64)).cuda(),
    )


def skyline_iteration_provider(netG, netD):
    real_label = 1
    fake_label = 0
    opt = dcgan.model_config()

    optimizerD = torch.optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    criterion = nn.BCELoss()

    device = torch.device("cuda")

    def iteration(*inputs):
        #for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        batch_size, data = inputs
        # train with real
        netD.zero_grad()
        real_cpu = data.to(device)
        label = torch.full((batch_size,), real_label,
                           dtype=real_cpu.dtype, device=device)
        output = netD(real_cpu)
        errD_real = criterion(output, label)
        errD_real.backward()

        # train with fake
        noise = torch.randn(batch_size, dcgan.nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        optimizerG.step()
    return iteration
