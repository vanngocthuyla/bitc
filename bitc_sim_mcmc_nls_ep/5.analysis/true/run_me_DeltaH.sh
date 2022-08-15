#!/bin/bash

export SCRIPT="/home/vla/python/bitc/bitc_sim_mcmc_nls_ep/scripts/run_plot_containing_rate_of_cis.py"

export MCMC_DIR="/home/vla/python/bitc/bitc_sim_mcmc_nls_ep/4.numpyro"

export EXP_DES_PAR_DIR="/home/vla/python/bitc/bitc_sim_mcmc_nls_ep/1.run_simulated_heats"

export EXPERIMENTS="0_sim 1_sim 2_sim 3_sim 4_sim 5_sim 6_sim 7_sim 8_sim 9_sim 10_sim 11_sim 12_sim 13_sim 14_sim 15_sim 16_sim 17_sim 18_sim 19_sim 20_sim 21_sim 22_sim 23_sim 24_sim 25_sim 26_sim 27_sim 28_sim 29_sim 30_sim 31_sim 32_sim 33_sim 34_sim 35_sim 36_sim 37_sim 38_sim 39_sim 40_sim 41_sim 42_sim 43_sim 44_sim 45_sim 46_sim 47_sim 48_sim 49_sim 50_sim 51_sim 52_sim 53_sim 54_sim 55_sim 56_sim 57_sim 58_sim 59_sim 60_sim 61_sim 62_sim 63_sim 64_sim 65_sim 66_sim 67_sim 68_sim 69_sim 70_sim 71_sim 72_sim 73_sim 74_sim 75_sim 76_sim 77_sim 78_sim 79_sim 80_sim 81_sim 82_sim 83_sim 84_sim 85_sim 86_sim 87_sim 88_sim 89_sim 90_sim 91_sim 92_sim 93_sim 94_sim 95_sim 96_sim 97_sim 98_sim 99_sim 100_sim 101_sim 102_sim 103_sim 104_sim 105_sim 106_sim 107_sim 108_sim 109_sim 110_sim 111_sim 112_sim 113_sim 114_sim 115_sim 116_sim 117_sim 118_sim 119_sim 120_sim 121_sim 122_sim 123_sim 124_sim 125_sim 126_sim 127_sim 128_sim 129_sim 130_sim 131_sim 132_sim 133_sim 134_sim 135_sim 136_sim 137_sim 138_sim 139_sim 140_sim 141_sim 142_sim 143_sim 144_sim 145_sim 146_sim 147_sim 148_sim 149_sim 150_sim 151_sim 152_sim 153_sim 154_sim 155_sim 156_sim 157_sim 158_sim 159_sim 160_sim 161_sim 162_sim 163_sim 164_sim 165_sim 166_sim 167_sim 168_sim 169_sim 170_sim 171_sim 172_sim 173_sim 174_sim 175_sim 176_sim 177_sim 178_sim 179_sim 180_sim 181_sim 182_sim 183_sim 184_sim 185_sim 186_sim 187_sim 188_sim 189_sim 190_sim 191_sim 192_sim 193_sim 194_sim 195_sim 196_sim 197_sim 198_sim 199_sim 200_sim 201_sim 202_sim 203_sim 204_sim 205_sim 206_sim 207_sim 208_sim 209_sim 210_sim 211_sim 212_sim 213_sim 214_sim 215_sim 216_sim 217_sim 218_sim 219_sim 220_sim 221_sim 222_sim 223_sim 224_sim 225_sim 226_sim 227_sim 228_sim 229_sim 230_sim 231_sim 232_sim 233_sim 234_sim 235_sim 236_sim 237_sim 238_sim 239_sim 240_sim 241_sim 242_sim 243_sim 244_sim 245_sim 246_sim 247_sim 248_sim 249_sim 250_sim 251_sim 252_sim 253_sim 254_sim 255_sim 256_sim 257_sim 258_sim 259_sim 260_sim 261_sim 262_sim 263_sim 264_sim 265_sim 266_sim 267_sim 268_sim 269_sim 270_sim 271_sim 272_sim 273_sim 274_sim 275_sim 276_sim 277_sim 278_sim 279_sim 280_sim 281_sim 282_sim 283_sim 284_sim 285_sim 286_sim 287_sim 288_sim 289_sim 290_sim 291_sim 292_sim 293_sim 294_sim 295_sim 296_sim 297_sim 298_sim 299_sim 300_sim 301_sim 302_sim 303_sim 304_sim 305_sim 306_sim 307_sim 308_sim 309_sim 310_sim 311_sim 312_sim 313_sim 314_sim 315_sim 316_sim 317_sim 318_sim 319_sim 320_sim 321_sim 322_sim 323_sim 324_sim 325_sim 326_sim 327_sim 328_sim 329_sim 330_sim 331_sim 332_sim 333_sim 334_sim 335_sim 336_sim 337_sim 338_sim 339_sim 340_sim 341_sim 342_sim 343_sim 344_sim 345_sim 346_sim 347_sim 348_sim 349_sim 350_sim 351_sim 352_sim 353_sim 354_sim 355_sim 356_sim 357_sim 358_sim 359_sim 360_sim 361_sim 362_sim 363_sim 364_sim 365_sim 366_sim 367_sim 368_sim 369_sim 370_sim 371_sim 372_sim 373_sim 374_sim 375_sim 376_sim 377_sim 378_sim 379_sim 380_sim 381_sim 382_sim 383_sim 384_sim 385_sim 386_sim 387_sim 388_sim 389_sim 390_sim 391_sim 392_sim 393_sim 394_sim 395_sim 396_sim 397_sim 398_sim 399_sim 400_sim 401_sim 402_sim 403_sim 404_sim 405_sim 406_sim 407_sim 408_sim 409_sim 410_sim 411_sim 412_sim 413_sim 414_sim 415_sim 416_sim 417_sim 418_sim 419_sim 420_sim 421_sim 422_sim 423_sim 424_sim 425_sim 426_sim 427_sim 428_sim 429_sim 430_sim 431_sim 432_sim 433_sim 434_sim 435_sim 436_sim 437_sim 438_sim 439_sim 440_sim 441_sim 442_sim 443_sim 444_sim 445_sim 446_sim 447_sim 448_sim 449_sim 450_sim 451_sim 452_sim 453_sim 454_sim 455_sim 456_sim 457_sim 458_sim 459_sim 460_sim 461_sim 462_sim 463_sim 464_sim 465_sim 466_sim 467_sim 468_sim 469_sim 470_sim 471_sim 472_sim 473_sim 474_sim 475_sim 476_sim 477_sim 478_sim 479_sim 480_sim 481_sim 482_sim 483_sim 484_sim 485_sim 486_sim 487_sim 488_sim 489_sim 490_sim 491_sim 492_sim 493_sim 494_sim 495_sim 496_sim 497_sim 498_sim 499_sim 500_sim 501_sim 502_sim 503_sim 504_sim 505_sim 506_sim 507_sim 508_sim 509_sim 510_sim 511_sim 512_sim 513_sim 514_sim 515_sim 516_sim 517_sim 518_sim 519_sim 520_sim 521_sim 522_sim 523_sim 524_sim 525_sim 526_sim 527_sim 528_sim 529_sim 530_sim 531_sim 532_sim 533_sim 534_sim 535_sim 536_sim 537_sim 538_sim 539_sim 540_sim 541_sim 542_sim 543_sim 544_sim 545_sim 546_sim 547_sim 548_sim 549_sim 550_sim 551_sim 552_sim 553_sim 554_sim 555_sim 556_sim 557_sim 558_sim 559_sim 560_sim 561_sim 562_sim 563_sim 564_sim 565_sim 566_sim 567_sim 568_sim 569_sim 570_sim 571_sim 572_sim 573_sim 574_sim 575_sim 576_sim 577_sim 578_sim 579_sim 580_sim 581_sim 582_sim 583_sim 584_sim 585_sim 586_sim 587_sim 588_sim 589_sim 590_sim 591_sim 592_sim 593_sim 594_sim 595_sim 596_sim 597_sim 598_sim 599_sim 600_sim 601_sim 602_sim 603_sim 604_sim 605_sim 606_sim 607_sim 608_sim 609_sim 610_sim 611_sim 612_sim 613_sim 614_sim 615_sim 616_sim 617_sim 618_sim 619_sim 620_sim 621_sim 622_sim 623_sim 624_sim 625_sim 626_sim 627_sim 628_sim 629_sim 630_sim 631_sim 632_sim 633_sim 634_sim 635_sim 636_sim 637_sim 638_sim 639_sim 640_sim 641_sim 642_sim 643_sim 644_sim 645_sim 646_sim 647_sim 648_sim 649_sim 650_sim 651_sim 652_sim 653_sim 654_sim 655_sim 656_sim 657_sim 658_sim 659_sim 660_sim 661_sim 662_sim 663_sim 664_sim 665_sim 666_sim 667_sim 668_sim 669_sim 670_sim 671_sim 672_sim 673_sim 674_sim 675_sim 676_sim 677_sim 678_sim 679_sim 680_sim 681_sim 682_sim 683_sim 684_sim 685_sim 686_sim 687_sim 688_sim 689_sim 690_sim 691_sim 692_sim 693_sim 694_sim 695_sim 696_sim 697_sim 698_sim 699_sim 700_sim 701_sim 702_sim 703_sim 704_sim 705_sim 706_sim 707_sim 708_sim 709_sim 710_sim 711_sim 712_sim 713_sim 714_sim 715_sim 716_sim 717_sim 718_sim 719_sim 720_sim 721_sim 722_sim 723_sim 724_sim 725_sim 726_sim 727_sim 728_sim 729_sim 730_sim 731_sim 732_sim 733_sim 734_sim 735_sim 736_sim 737_sim 738_sim 739_sim 740_sim 741_sim 742_sim 743_sim 744_sim 745_sim 746_sim 747_sim 748_sim 749_sim 750_sim 751_sim 752_sim 753_sim 754_sim 755_sim 756_sim 757_sim 758_sim 759_sim 760_sim 761_sim 762_sim 763_sim 764_sim 765_sim 766_sim 767_sim 768_sim 769_sim 770_sim 771_sim 772_sim 773_sim 774_sim 775_sim 776_sim 777_sim 778_sim 779_sim 780_sim 781_sim 782_sim 783_sim 784_sim 785_sim 786_sim 787_sim 788_sim 789_sim 790_sim 791_sim 792_sim 793_sim 794_sim 795_sim 796_sim 797_sim 798_sim 799_sim 800_sim 801_sim 802_sim 803_sim 804_sim 805_sim 806_sim 807_sim 808_sim 809_sim 810_sim 811_sim 812_sim 813_sim 814_sim 815_sim 816_sim 817_sim 818_sim 819_sim 820_sim 821_sim 822_sim 823_sim 824_sim 825_sim 826_sim 827_sim 828_sim 829_sim 830_sim 831_sim 832_sim 833_sim 834_sim 835_sim 836_sim 837_sim 838_sim 839_sim 840_sim 841_sim 842_sim 843_sim 844_sim 845_sim 846_sim 847_sim 848_sim 849_sim 850_sim 851_sim 852_sim 853_sim 854_sim 855_sim 856_sim 857_sim 858_sim 859_sim 860_sim 861_sim 862_sim 863_sim 864_sim 865_sim 866_sim 867_sim 868_sim 869_sim 870_sim 871_sim 872_sim 873_sim 874_sim 875_sim 876_sim 877_sim 878_sim 879_sim 880_sim 881_sim 882_sim 883_sim 884_sim 885_sim 886_sim 887_sim 888_sim 889_sim 890_sim 891_sim 892_sim 893_sim 894_sim 895_sim 896_sim 897_sim 898_sim 899_sim 900_sim 901_sim 902_sim 903_sim 904_sim 905_sim 906_sim 907_sim 908_sim 909_sim 910_sim 911_sim 912_sim 913_sim 914_sim 915_sim 916_sim 917_sim 918_sim 919_sim 920_sim 921_sim 922_sim 923_sim 924_sim 925_sim 926_sim 927_sim 928_sim 929_sim 930_sim 931_sim 932_sim 933_sim 934_sim 935_sim 936_sim 937_sim 938_sim 939_sim 940_sim 941_sim 942_sim 943_sim 944_sim 945_sim 946_sim 947_sim 948_sim 949_sim 950_sim 951_sim 952_sim 953_sim 954_sim 955_sim 956_sim 957_sim 958_sim 959_sim 960_sim 961_sim 962_sim 963_sim 964_sim 965_sim 966_sim 967_sim 968_sim 969_sim 970_sim 971_sim 972_sim 973_sim 974_sim 975_sim 976_sim 977_sim 978_sim 979_sim 980_sim 981_sim 982_sim 983_sim 984_sim 985_sim 986_sim 987_sim 988_sim 989_sim 990_sim 991_sim 992_sim 993_sim 994_sim 995_sim 996_sim 997_sim 998_sim 999_sim"

export CENTRAL="median"

# DeltaH
python $SCRIPT --bitc_mcmc_dir $MCMC_DIR --ordered_experiment_names "$EXPERIMENTS" --central $CENTRAL --parameter "DeltaH" --experimental_design_parameters_dir $EXP_DES_PAR_DIR
