montage ./test_xs_img.jpg ./test_xs_obj.jpg -bordercolor white -borderwidth 0 -geometry +40+1 -tile 2x1 -resize 400x200 rec_target.jpg

montage ./img_recon_sm_fcn_corrup_noise/reconstruction_00000025.png ./img_recon_sm_fcn_corrup_noise/reconstruction_00000050.png ./img_recon_sm_fcn_corrup_noise/reconstruction_00000100.png ./img_recon_sm_fcn_corrup_noise/reconstruction_00000200.png ./img_recon_sm_fcn_corrup_noise/reconstruction_00000800.png -bordercolor white -borderwidth 0 -geometry +4+1 -tile 5x1 -resize 1000x200 rec_corrup_noise_fcn.jpg

montage ./img_recon_sm_fcn/reconstruction_00000025.png ./img_recon_sm_fcn/reconstruction_00000050.png ./img_recon_sm_fcn/reconstruction_00000100.png ./img_recon_sm_fcn/reconstruction_00000200.png ./img_recon_sm_fcn/reconstruction_00000780.png -bordercolor white -borderwidth 0 -geometry +4+1 -tile 5x1 -resize 1000x200 rec_fcn.jpg

montage ./img_recon_sm_corrup_ALL/reconstruction_00000025.png ./img_recon_sm_corrup_ALL/reconstruction_00000050.png ./img_recon_sm_corrup_ALL/reconstruction_00000100.png ./img_recon_sm_corrup_ALL/reconstruction_00000200.png ./img_recon_sm_corrup_ALL/reconstruction_00000537.png -bordercolor white -borderwidth 0 -geometry +4+1 -tile 5x1 -resize 1000x200 rec_corrupALL.jpg

montage ./img_recon_cvae/reconstruction_00000025.png ./img_recon_cvae/reconstruction_00000050.png ./img_recon_cvae/reconstruction_00000100.png ./img_recon_cvae/reconstruction_00000200.png ./img_recon_cvae/reconstruction_00002500.png -bordercolor white -borderwidth 0 -geometry +4+1 -tile 5x1 -resize 1000x200 rec_cvae.jpg
