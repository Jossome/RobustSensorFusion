adversary = dict(
        sensor='image',
        method='pgd',
        adv_loss='loss_bbox',
        iters=20,
        alpha_img=4/255,
        alpha_pts=4/255,
        eps_img=0.05,
        eps_pts=0.05,
        norm='l_inf',
        restrict_region=False,
        patch_size=0.5,
        random_keep=0.5) # At what chance will we keep the original input without perturbing it
