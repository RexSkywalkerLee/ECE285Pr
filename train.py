import torch
import torch
import itertools
from dataloader import UnalignedDataset
from model import *

def train(epoch_start=150, n_epoch=100, n_epoch_decay=100, batch_size=4, hed=True, dev='cuda:0'):
    dataset = UnalignedDataset()
    dataset_size = len(dataset)
    print('The number of training images = %d' % dataset_size)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=4,
                                             pin_memory=True)
    
    device = torch.device(dev)
    
    l_channel = 3
    if hed:
        p_channel = 4
    else:
        p_channel = 3
        
    p2l_G = Generator(input_channel=p_channel, output_channel=l_channel)
    l2p_G = Generator(input_channel=l_channel, output_channel=p_channel)
    p_D = Discriminator(input_channel=p_channel)
    l_D = Discriminator(input_channel=l_channel)
    
    if hed:
        hed_net = HED()
        hed_net.load_state_dict(torch.load('./pretrained_hed_net.pth'))
        for param in hed_net.parameters():
            param.requires_grad = False
        hed_net = hed_net.to(device)
    
    if epoch_start != 1:
        p2l_G.load_state_dict(torch.load('./networks/cyclegan_cathed/p2l_G_net.pth'))
        l2p_G.load_state_dict(torch.load('./networks/cyclegan_cathed/l2p_G_net.pth'))
        p_D.load_state_dict(torch.load('./networks/cyclegan_cathed/p_D_net.pth'))
        l_D.load_state_dict(torch.load('./networks/cyclegan_cathed/l_D_net.pth'))
        
    p2l_G = p2l_G.to(device)
    l2p_G = l2p_G.to(device)
    p_D = p_D.to(device)
    l_D = l_D.to(device)
    
    lr = 0.0002
    optimizer_G = torch.optim.Adam(itertools.chain(p2l_G.parameters(), l2p_G.parameters()), lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(itertools.chain(p_D.parameters(), l_D.parameters()), lr=lr, betas=(0.5, 0.999))
    
    def lambda_rule(epoch):
        lr_l = 1 - max(0,epoch+1-100) / (100+1)
        return lr_l
    scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda_rule)
    scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda_rule)

    for epoch in range(epoch_start, n_epoch+n_epoch_decay+1):
        for i, data in enumerate(dataloader):
            real_l, real_p = data
            real_l = real_l.to(device)
            real_p = real_p.to(device)
            if hed:
                real_p = torch.cat([real_p, torch.sigmoid(hed_net(real_p))], dim=1)
            fake_l = p2l_G(real_p)
            fake_p = l2p_G(real_l)
            rec_l = p2l_G(fake_p)
            rec_p = l2p_G(fake_l)
            if hed:
                idt_l = p2l_G(torch.cat([real_l, torch.sigmoid(hed_net(real_l))], dim=1))
                idt_p = l2p_G(real_p[:,0:3,:,:])
                # idt_p = l2p_G(real_p.repeat(1,3,1,1))
            else:
                idt_l = p2l_G(real_l)
                idt_p = l2p_G(real_p)
            
            # Optimize generators
            for param in p_D.parameters():
                param.requires_grad = False
            for param in l_D.parameters():
                param.requires_grad = False
            optimizer_G.zero_grad()

            crit_gan = torch.nn.MSELoss()
            pred_p = p_D(fake_p)
            target_p = torch.ones_like(pred_p).to(pred_p.device)
            pred_l = l_D(fake_l)
            target_l = torch.ones_like(pred_l).to(pred_l.device)
            gan_loss = crit_gan(pred_p, target_p) + crit_gan(pred_l, target_l)
            
            crit_cyc = torch.nn.L1Loss()
            cyc_loss = crit_cyc(rec_l, real_l) + crit_cyc(rec_p, real_p)
            
            crit_idt = torch.nn.L1Loss()
            idt_loss = crit_idt(idt_l, real_l) + crit_idt(idt_p, real_p)
            
            loss_G = gan_loss + 10 * cyc_loss + 5 * idt_loss
            loss_G.backward()
            optimizer_G.step()
            
            # Optimize discriminators
            for param in p_D.parameters():
                param.requires_grad = True
            for param in l_D.parameters():
                param.requires_grad = True
            optimizer_D.zero_grad()
            
            crit_dis = torch.nn.MSELoss()
            
            pred_real_p = p_D(real_p)
            target_real_p = torch.ones_like(pred_real_p).to(pred_real_p.device)
            pred_fake_p = p_D(fake_p.detach())
            target_fake_p = torch.zeros_like(pred_fake_p).to(pred_fake_p.device)
            loss_p = crit_dis(pred_real_p, target_real_p) + crit_dis(pred_fake_p, target_fake_p)
            
            pred_real_l = l_D(real_l)
            target_real_l = torch.ones_like(pred_real_l).to(pred_real_l.device)
            pred_fake_l = l_D(fake_l.detach())
            target_fake_l = torch.zeros_like(pred_fake_l).to(pred_fake_l.device)
            loss_l = crit_dis(pred_real_l, target_real_l) + crit_dis(pred_fake_l, target_fake_l)
            
            loss_D = 0.5 * (loss_p + loss_l)
            loss_D.backward()
            optimizer_D.step()
            
            print('losses at iter [%.3f/1] of epoch %d: loss_G: %.3f loss_D: %.3f' % (i*batch_size/dataset_size, epoch, loss_G.cpu().item(), loss_D.cpu().item()))
            
        scheduler_G.step()
        scheduler_D.step()
        
        print('saving the model at the end of epoch %d' % epoch)
        torch.save(p2l_G.state_dict(), 'networks/cyclegan_cathed/p2l_G_net.pth')
        torch.save(l2p_G.state_dict(), 'networks/cyclegan_cathed/l2p_G_net.pth')
        torch.save(p_D.state_dict(), 'networks/cyclegan_cathed/p_D_net.pth')
        torch.save(l_D.state_dict(), 'networks/cyclegan_cathed/l_D_net.pth')

if __name__ == '__main__':
    train()