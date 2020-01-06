# Probabilistic UNet in PyTorch
A Probabilistic U-Net for segmentation of ambiguous images implemented in PyTorch. This is a pytorch implementation of this paper https://arxiv.org/abs/1806.05034, for which the code can be found here: https://github.com/SimonKohl/probabilistic_unet. 

## Adding KL divergence for Independent distribution
In order to implement an Gaussian distribution with an axis aligned covariance matrix in PyTorch, I needed to wrap a Normal distribution in a Independent distribution. Therefore you need the add the following to the PyTorch source code at torch/distributions/kl.py (source: https://github.com/pytorch/pytorch/issues/13545).

```
def _kl_independent_independent(p, q):
    if p.reinterpreted_batch_ndims != q.reinterpreted_batch_ndims:
        raise NotImplementedError
    result = kl_divergence(p.base_dist, q.base_dist)
    return _sum_rightmost(result, p.reinterpreted_batch_ndims)
```

## Training
In order to train your own Probabilistic UNet in PyTorch, you should first write your own data loader. Then you can use the following code snippet to train the network

```
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
```

## Train on LIDC Dataset
One of the datasets used in the original paper is the [LIDC dataset](https://wiki.cancerimagingarchive.net). I've preprocessed this data and stored them in a pickle file, which you can [download here](https://drive.google.com/drive/folders/1xKfKCQo8qa6SAr3u7qWNtQjIphIrvmd5?usp=sharing). After downloading the files you should place them in a folder called 'data'. After that, you can train your own Probabilistic UNet on the LIDC dataset using the simple train script provided in train_model.py.



# PyTorch中的概率UNet
在PyTorch中实现的用于分割模糊图像的概率U网络。这是本文https://arxiv.org/abs/1806.05034的pytorch实现，其代码可以在这里找到：https://github.com/SimonKohl/probabilistic_unet。

## 添加KL散度以进行独立分布
为了在PyTorch中使用与轴对齐的协方差矩阵实现高斯分布，我需要将正态分布包装为独立分布。因此，您需要在torch/distributions/kl.py （源：https://github.com/pytorch/pytorch/issues/13545）上将以下内容添加到PyTorch源代码中。

```
def _kl_independent_independent(p, q):
    if p.reinterpreted_batch_ndims != q.reinterpreted_batch_ndims:
        raise NotImplementedError
    result = kl_divergence(p.base_dist, q.base_dist)
    return _sum_rightmost(result, p.reinterpreted_batch_ndims)
```

## 训练
为了在PyTorch中训练自己的概率UNet，您应该首先编写自己的数据加载器。然后，您可以使用以下代码段来训练网络

```
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
```

## 在LIDC数据集上训练
原始论文中使用的数据集之一是[LIDC数据集]（https://wiki.cancerimagingarchive.net）。我已经对该数据进行了预处理，并将它们存储在一个pickle文件中，您可以在[下载此处]（https://drive.google.com/drive/folders/1xKfKCQo8qa6SAr3u7qWNtQjIphIrvmd5?usp=sharing）中进行下载。下载文件后，应将它们放置在名为“ data”的文件夹中。之后，您可以使用train_model.py中提供的简单训练脚本在LIDC数据集上训练自己的概率UNet。

