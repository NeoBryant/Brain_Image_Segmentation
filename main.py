# coding=utf-8

from unet import Unet


train_loader = define this yourself
net = ProbabilisticUnet(no_channels,no_classes,filter_list,latent_dim,no_fcomb_convs,beta)
net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=0)
for epoch in range(epochs):
    for step, (patch, mask) in enumerate(train_loader): 
        patch = patch.to(device)
        mask = mask.to(device)
        mask = torch.unsqueeze(mask,1)
        net.forward(patch, mask, training=True)
        elbo = net.elbo(mask)
        reg_loss = l2_regularisation(net.posterior) + l2_regularisation(net.prior) + l2_regularisation(net.fcomb.layers)
        loss = -elbo + 1e-5 * reg_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()