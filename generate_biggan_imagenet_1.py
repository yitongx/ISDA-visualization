'''
ISDA训练GAN噪声进行可视化代码
改成了Biggan
'''
import time, argparse, datetime
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import save_images
import torchvision.transforms as transforms
import torch.optim as optim
import os
import numpy as np
import random
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample, save_as_images, display_in_terminal)
from PIL import Image
import networks_imagenet.resnet
from ISDA_imagenet import ISDALoss

parser = argparse.ArgumentParser(description='BigGAN visualization')
parser.add_argument('--print_freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--epoch1', default=5000, type=int, metavar='E1', help='step1 epoch')
parser.add_argument('--epoch2', default=5000, type=int, metavar='E2', help='step2 epoch')
parser.add_argument('--lr1', default=10, type=float, metavar='LR1', help='LR1')
parser.add_argument('--lr2', default=1, type=float, metavar='LR2', help='LR2')
parser.add_argument('--truncation', default=0.4, type=float, metavar='TRUNC', help='truncation for BigGAN noise vector, default=0.4')
parser.add_argument('--noise_seed', default=1, type=int, metavar='NS', help='Seed for noise vector')

parser.add_argument('--aug_number', default=32, type=int, metavar='AN', help='number of augment results for one real image')
parser.add_argument('--aug_alpha', default=0.2, type=float, metavar='AA', help='For feature augmentation control')
parser.add_argument('--img_dir', default='imagenet/n02106030_239.JPEG', \
                    type=str, metavar='ID', help='class label number in ImageNet')
parser.add_argument('--name', default='generator', type=str, help='name of experiment')
parser.add_argument('--job_id', default='', type=str, help='shell id')
parser.add_argument('--cloud', default=True, type=bool, help='whether to use cloud')


parser.add_argument('--data_url', default='./', type=str, help='root to train data')
parser.add_argument('--train_url', default='./', type=str, help='for huawei cloud')
parser.add_argument('--init_method', default='', type=str, help='for huawei cloud')

args = parser.parse_args()
print(args)

import moxing as mox
mox.file.shift('os', 'mox')

global mask_train_mean
global mask_train_std
global mask_test_mean
global mask_test_std

class_num = 1000
feature_num = 2048

mean_mask = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).expand(1, 3, 512, 512).cuda()
std_mask = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).expand(1, 3, 512, 512).cuda()

IMAGENET = {1440764: 0, 1443537: 1, 1484850: 2, 1491361: 3, 1494475: 4, 1496331: 5, 1498041: 6, 1514668: 7, 1514859: 8, 1518878: 9, 1530575: 10, 1531178: 11, 1532829: 12, 1534433: 13, 1537544: 14, 1558993: 15, 1560419: 16, 1580077: 17, 1582220: 18, 1592084: 19, 1601694: 20, 1608432: 21, 1614925: 22, 1616318: 23, 1622779: 24, 1629819: 25, 1630670: 26, 1631663: 27, 1632458: 28, 1632777: 29, 1641577: 30, 1644373: 31, 1644900: 32, 1664065: 33, 1665541: 34, 1667114: 35, 1667778: 36, 1669191: 37, 1675722: 38, 1677366: 39, 1682714: 40, 1685808: 41, 1687978: 42, 1688243: 43, 1689811: 44, 1692333: 45, 1693334: 46, 1694178: 47, 1695060: 48, 1697457: 49, 1698640: 50, 1704323: 51, 1728572: 52, 1728920: 53, 1729322: 54, 1729977: 55, 1734418: 56, 1735189: 57, 1737021: 58, 1739381: 59, 1740131: 60, 1742172: 61, 1744401: 62, 1748264: 63, 1749939: 64, 1751748: 65, 1753488: 66, 1755581: 67, 1756291: 68, 1768244: 69, 1770081: 70, 1770393: 71, 1773157: 72, 1773549: 73, 1773797: 74, 1774384: 75, 1774750: 76, 1775062: 77, 1776313: 78, 1784675: 79, 1795545: 80, 1796340: 81, 1797886: 82, 1798484: 83, 1806143: 84, 1806567: 85, 1807496: 86, 1817953: 87, 1818515: 88, 1819313: 89, 1820546: 90, 1824575: 91, 1828970: 92, 1829413: 93, 1833805: 94, 1843065: 95, 1843383: 96, 1847000: 97, 1855032: 98, 1855672: 99, 1860187: 100, 1871265: 101, 1872401: 102, 1873310: 103, 1877812: 104, 1882714: 105, 1883070: 106, 1910747: 107, 1914609: 108, 1917289: 109, 1924916: 110, 1930112: 111, 1943899: 112, 1944390: 113, 1945685: 114, 1950731: 115, 1955084: 116, 1968897: 117, 1978287: 118, 1978455: 119, 1980166: 120, 1981276: 121, 1983481: 122, 1984695: 123, 1985128: 124, 1986214: 125, 1990800: 126, 2002556: 127, 2002724: 128, 2006656: 129, 2007558: 130, 2009229: 131, 2009912: 132, 2011460: 133, 2012849: 134, 2013706: 135, 2017213: 136, 2018207: 137, 2018795: 138, 2025239: 139, 2027492: 140, 2028035: 141, 2033041: 142, 2037110: 143, 2051845: 144, 2056570: 145, 2058221: 146, 2066245: 147, 2071294: 148, 2074367: 149, 2077923: 150, 2085620: 151, 2085782: 152, 2085936: 153, 2086079: 154, 2086240: 155, 2086646: 156, 2086910: 157, 2087046: 158, 2087394: 159, 2088094: 160, 2088238: 161, 2088364: 162, 2088466: 163, 2088632: 164, 2089078: 165, 2089867: 166, 2089973: 167, 2090379: 168, 2090622: 169, 2090721: 170, 2091032: 171, 2091134: 172, 2091244: 173, 2091467: 174, 2091635: 175, 2091831: 176, 2092002: 177, 2092339: 178, 2093256: 179, 2093428: 180, 2093647: 181, 2093754: 182, 2093859: 183, 2093991: 184, 2094114: 185, 2094258: 186, 2094433: 187, 2095314: 188, 2095570: 189, 2095889: 190, 2096051: 191, 2096177: 192, 2096294: 193, 2096437: 194, 2096585: 195, 2097047: 196, 2097130: 197, 2097209: 198, 2097298: 199, 2097474: 200, 2097658: 201, 2098105: 202, 2098286: 203, 2098413: 204, 2099267: 205, 2099429: 206, 2099601: 207, 2099712: 208, 2099849: 209, 2100236: 210, 2100583: 211, 2100735: 212, 2100877: 213, 2101006: 214, 2101388: 215, 2101556: 216, 2102040: 217, 2102177: 218, 2102318: 219, 2102480: 220, 2102973: 221, 2104029: 222, 2104365: 223, 2105056: 224, 2105162: 225, 2105251: 226, 2105412: 227, 2105505: 228, 2105641: 229, 2105855: 230, 2106030: 231, 2106166: 232, 2106382: 233, 2106550: 234, 2106662: 235, 2107142: 236, 2107312: 237, 2107574: 238, 2107683: 239, 2107908: 240, 2108000: 241, 2108089: 242, 2108422: 243, 2108551: 244, 2108915: 245, 2109047: 246, 2109525: 247, 2109961: 248, 2110063: 249, 2110185: 250, 2110341: 251, 2110627: 252, 2110806: 253, 2110958: 254, 2111129: 255, 2111277: 256, 2111500: 257, 2111889: 258, 2112018: 259, 2112137: 260, 2112350: 261, 2112706: 262, 2113023: 263, 2113186: 264, 2113624: 265, 2113712: 266, 2113799: 267, 2113978: 268, 2114367: 269, 2114548: 270, 2114712: 271, 2114855: 272, 2115641: 273, 2115913: 274, 2116738: 275, 2117135: 276, 2119022: 277, 2119789: 278, 2120079: 279, 2120505: 280, 2123045: 281, 2123159: 282, 2123394: 283, 2123597: 284, 2124075: 285, 2125311: 286, 2127052: 287, 2128385: 288, 2128757: 289, 2128925: 290, 2129165: 291, 2129604: 292, 2130308: 293, 2132136: 294, 2133161: 295, 2134084: 296, 2134418: 297, 2137549: 298, 2138441: 299, 2165105: 300, 2165456: 301, 2167151: 302, 2168699: 303, 2169497: 304, 2172182: 305, 2174001: 306, 2177972: 307, 2190166: 308, 2206856: 309, 2219486: 310, 2226429: 311, 2229544: 312, 2231487: 313, 2233338: 314, 2236044: 315, 2256656: 316, 2259212: 317, 2264363: 318, 2268443: 319, 2268853: 320, 2276258: 321, 2277742: 322, 2279972: 323, 2280649: 324, 2281406: 325, 2281787: 326, 2317335: 327, 2319095: 328, 2321529: 329, 2325366: 330, 2326432: 331, 2328150: 332, 2342885: 333, 2346627: 334, 2356798: 335, 2361337: 336, 2363005: 337, 2364673: 338, 2389026: 339, 2391049: 340, 2395406: 341, 2396427: 342, 2397096: 343, 2398521: 344, 2403003: 345, 2408429: 346, 2410509: 347, 2412080: 348, 2415577: 349, 2417914: 350, 2422106: 351, 2422699: 352, 2423022: 353, 2437312: 354, 2437616: 355, 2441942: 356, 2442845: 357, 2443114: 358, 2443484: 359, 2444819: 360, 2445715: 361, 2447366: 362, 2454379: 363, 2457408: 364, 2480495: 365, 2480855: 366, 2481823: 367, 2483362: 368, 2483708: 369, 2484975: 370, 2486261: 371, 2486410: 372, 2487347: 373, 2488291: 374, 2488702: 375, 2489166: 376, 2490219: 377, 2492035: 378, 2492660: 379, 2493509: 380, 2493793: 381, 2494079: 382, 2497673: 383, 2500267: 384, 2504013: 385, 2504458: 386, 2509815: 387, 2510455: 388, 2514041: 389, 2526121: 390, 2536864: 391, 2606052: 392, 2607072: 393, 2640242: 394, 2641379: 395, 2643566: 396, 2655020: 397, 2666196: 398, 2667093: 399, 2669723: 400, 2672831: 401, 2676566: 402, 2687172: 403, 2690373: 404, 2692877: 405, 2699494: 406, 2701002: 407, 2704792: 408, 2708093: 409, 2727426: 410, 2730930: 411, 2747177: 412, 2749479: 413, 2769748: 414, 2776631: 415, 2777292: 416, 2782093: 417, 2783161: 418, 2786058: 419, 2787622: 420, 2788148: 421, 2790996: 422, 2791124: 423, 2791270: 424, 2793495: 425, 2794156: 426, 2795169: 427, 2797295: 428, 2799071: 429, 2802426: 430, 2804414: 431, 2804610: 432, 2807133: 433, 2808304: 434, 2808440: 435, 2814533: 436, 2814860: 437, 2815834: 438, 2817516: 439, 2823428: 440, 2823750: 441, 2825657: 442, 2834397: 443, 2835271: 444, 2837789: 445, 2840245: 446, 2841315: 447, 2843684: 448, 2859443: 449, 2860847: 450, 2865351: 451, 2869837: 452, 2870880: 453, 2871525: 454, 2877765: 455, 2879718: 456, 2883205: 457, 2892201: 458, 2892767: 459, 2894605: 460, 2895154: 461, 2906734: 462, 2909870: 463, 2910353: 464, 2916936: 465, 2917067: 466, 2927161: 467, 2930766: 468, 2939185: 469, 2948072: 470, 2950826: 471, 2951358: 472, 2951585: 473, 2963159: 474, 2965783: 475, 2966193: 476, 2966687: 477, 2971356: 478, 2974003: 479, 2977058: 480, 2978881: 481, 2979186: 482, 2980441: 483, 2981792: 484, 2988304: 485, 2992211: 486, 2992529: 487, 2999410: 488, 3000134: 489, 3000247: 490, 3000684: 491, 3014705: 492, 3016953: 493, 3017168: 494, 3018349: 495, 3026506: 496, 3028079: 497, 3032252: 498, 3041632: 499, 3042490: 500, 3045698: 501, 3047690: 502, 3062245: 503, 3063599: 504, 3063689: 505, 3065424: 506, 3075370: 507, 3085013: 508, 3089624: 509, 3095699: 510, 3100240: 511, 3109150: 512, 3110669: 513, 3124043: 514, 3124170: 515, 3125729: 516, 3126707: 517, 3127747: 518, 3127925: 519, 3131574: 520, 3133878: 521, 3134739: 522, 3141823: 523, 3146219: 524, 3160309: 525, 3179701: 526, 3180011: 527, 3187595: 528, 3188531: 529, 3196217: 530, 3197337: 531, 3201208: 532, 3207743: 533, 3207941: 534, 3208938: 535, 3216828: 536, 3218198: 537, 3220513: 538, 3223299: 539, 3240683: 540, 3249569: 541, 3250847: 542, 3255030: 543, 3259280: 544, 3271574: 545, 3272010: 546, 3272562: 547, 3290653: 548, 3291819: 549, 3297495: 550, 3314780: 551, 3325584: 552, 3337140: 553, 3344393: 554, 3345487: 555, 3347037: 556, 3355925: 557, 3372029: 558, 3376595: 559, 3379051: 560, 3384352: 561, 3388043: 562, 3388183: 563, 3388549: 564, 3393912: 565, 3394916: 566, 3400231: 567, 3404251: 568, 3417042: 569, 3424325: 570, 3425413: 571, 3443371: 572, 3444034: 573, 3445777: 574, 3445924: 575, 3447447: 576, 3447721: 577, 3450230: 578, 3452741: 579, 3457902: 580, 3459775: 581, 3461385: 582, 3467068: 583, 3476684: 584, 3476991: 585, 3478589: 586, 3481172: 587, 3482405: 588, 3483316: 589, 3485407: 590, 3485794: 591, 3492542: 592, 3494278: 593, 3495258: 594, 3496892: 595, 3498962: 596, 3527444: 597, 3529860: 598, 3530642: 599, 3532672: 600, 3534580: 601, 3535780: 602, 3538406: 603, 3544143: 604, 3584254: 605, 3584829: 606, 3590841: 607, 3594734: 608, 3594945: 609, 3595614: 610, 3598930: 611, 3599486: 612, 3602883: 613, 3617480: 614, 3623198: 615, 3627232: 616, 3630383: 617, 3633091: 618, 3637318: 619, 3642806: 620, 3649909: 621, 3657121: 622, 3658185: 623, 3661043: 624, 3662601: 625, 3666591: 626, 3670208: 627, 3673027: 628, 3676483: 629, 3680355: 630, 3690938: 631, 3691459: 632, 3692522: 633, 3697007: 634, 3706229: 635, 3709823: 636, 3710193: 637, 3710637: 638, 3710721: 639, 3717622: 640, 3720891: 641, 3721384: 642, 3724870: 643, 3729826: 644, 3733131: 645, 3733281: 646, 3733805: 647, 3742115: 648, 3743016: 649, 3759954: 650, 3761084: 651, 3763968: 652, 3764736: 653, 3769881: 654, 3770439: 655, 3770679: 656, 3773504: 657, 3775071: 658, 3775546: 659, 3776460: 660, 3777568: 661, 3777754: 662, 3781244: 663, 3782006: 664, 3785016: 665, 3786901: 666, 3787032: 667, 3788195: 668, 3788365: 669, 3791053: 670, 3792782: 671, 3792972: 672, 3793489: 673, 3794056: 674, 3796401: 675, 3803284: 676, 3804744: 677, 3814639: 678, 3814906: 679, 3825788: 680, 3832673: 681, 3837869: 682, 3838899: 683, 3840681: 684, 3841143: 685, 3843555: 686, 3854065: 687, 3857828: 688, 3866082: 689, 3868242: 690, 3868863: 691, 3871628: 692, 3873416: 693, 3874293: 694, 3874599: 695, 3876231: 696, 3877472: 697, 3877845: 698, 3884397: 699, 3887697: 700, 3888257: 701, 3888605: 702, 3891251: 703, 3891332: 704, 3895866: 705, 3899768: 706, 3902125: 707, 3903868: 708, 3908618: 709, 3908714: 710, 3916031: 711, 3920288: 712, 3924679: 713, 3929660: 714, 3929855: 715, 3930313: 716, 3930630: 717, 3933933: 718, 3935335: 719, 3937543: 720, 3938244: 721, 3942813: 722, 3944341: 723, 3947888: 724, 3950228: 725, 3954731: 726, 3956157: 727, 3958227: 728, 3961711: 729, 3967562: 730, 3970156: 731, 3976467: 732, 3976657: 733, 3977966: 734, 3980874: 735, 3982430: 736, 3983396: 737, 3991062: 738, 3992509: 739, 3995372: 740, 3998194: 741, 4004767: 742, 4005630: 743, 4008634: 744, 4009552: 745, 4019541: 746, 4023962: 747, 4026417: 748, 4033901: 749, 4033995: 750, 4037443: 751, 4039381: 752, 4040759: 753, 4041544: 754, 4044716: 755, 4049303: 756, 4065272: 757, 4067472: 758, 4069434: 759, 4070727: 760, 4074963: 761, 4081281: 762, 4086273: 763, 4090263: 764, 4099969: 765, 4111531: 766, 4116512: 767, 4118538: 768, 4118776: 769, 4120489: 770, 4125021: 771, 4127249: 772, 4131690: 773, 4133789: 774, 4136333: 775, 4141076: 776, 4141327: 777, 4141975: 778, 4146614: 779, 4147183: 780, 4149813: 781, 4152593: 782, 4153751: 783, 4154565: 784, 4162706: 785, 4179913: 786, 4192698: 787, 4200800: 788, 4201297: 789, 4204238: 790, 4204347: 791, 4208210: 792, 4209133: 793, 4209239: 794, 4228054: 795, 4229816: 796, 4235860: 797, 4238763: 798, 4239074: 799, 4243546: 800, 4251144: 801, 4252077: 802, 4252225: 803, 4254120: 804, 4254680: 805, 4254777: 806, 4258138: 807, 4259630: 808, 4263257: 809, 4264628: 810, 4265275: 811, 4266014: 812, 4270147: 813, 4273569: 814, 4275548: 815, 4277352: 816, 4285008: 817, 4286575: 818, 4296562: 819, 4310018: 820, 4311004: 821, 4311174: 822, 4317175: 823, 4325704: 824, 4326547: 825, 4328186: 826, 4330267: 827, 4332243: 828, 4335435: 829, 4336792: 830, 4344873: 831, 4346328: 832, 4347754: 833, 4350905: 834, 4355338: 835, 4355933: 836, 4356056: 837, 4357314: 838, 4366367: 839, 4367480: 840, 4370456: 841, 4371430: 842, 4371774: 843, 4372370: 844, 4376876: 845, 4380533: 846, 4389033: 847, 4392985: 848, 4398044: 849, 4399382: 850, 4404412: 851, 4409515: 852, 4417672: 853, 4418357: 854, 4423845: 855, 4428191: 856, 4429376: 857, 4435653: 858, 4442312: 859, 4443257: 860, 4447861: 861, 4456115: 862, 4458633: 863, 4461696: 864, 4462240: 865, 4465501: 866, 4467665: 867, 4476259: 868, 4479046: 869, 4482393: 870, 4483307: 871, 4485082: 872, 4486054: 873, 4487081: 874, 4487394: 875, 4493381: 876, 4501370: 877, 4505470: 878, 4507155: 879, 4509417: 880, 4515003: 881, 4517823: 882, 4522168: 883, 4523525: 884, 4525038: 885, 4525305: 886, 4532106: 887, 4532670: 888, 4536866: 889, 4540053: 890, 4542943: 891, 4548280: 892, 4548362: 893, 4550184: 894, 4552348: 895, 4553703: 896, 4554684: 897, 4557648: 898, 4560804: 899, 4562935: 900, 4579145: 901, 4579432: 902, 4584207: 903, 4589890: 904, 4590129: 905, 4591157: 906, 4591713: 907, 4592741: 908, 4596742: 909, 4597913: 910, 4599235: 911, 4604644: 912, 4606251: 913, 4612504: 914, 4613696: 915, 6359193: 916, 6596364: 917, 6785654: 918, 6794110: 919, 6874185: 920, 7248320: 921, 7565083: 922, 7579787: 923, 7583066: 924, 7584110: 925, 7590611: 926, 7613480: 927, 7614500: 928, 7615774: 929, 7684084: 930, 7693725: 931, 7695742: 932, 7697313: 933, 7697537: 934, 7711569: 935, 7714571: 936, 7714990: 937, 7715103: 938, 7716358: 939, 7716906: 940, 7717410: 941, 7717556: 942, 7718472: 943, 7718747: 944, 7720875: 945, 7730033: 946, 7734744: 947, 7742313: 948, 7745940: 949, 7747607: 950, 7749582: 951, 7753113: 952, 7753275: 953, 7753592: 954, 7754684: 955, 7760859: 956, 7768694: 957, 7802026: 958, 7831146: 959, 7836838: 960, 7860988: 961, 7871810: 962, 7873807: 963, 7875152: 964, 7880968: 965, 7892512: 966, 7920052: 967, 7930864: 968, 7932039: 969, 9193705: 970, 9229709: 971, 9246464: 972, 9256479: 973, 9288635: 974, 9332890: 975, 9399592: 976, 9421951: 977, 9428293: 978, 9468604: 979, 9472597: 980, 9835506: 981, 10148035: 982, 10565667: 983, 11879895: 984, 11939491: 985, 12057211: 986, 12144580: 987, 12267677: 988, 12620546: 989, 12768682: 990, 12985857: 991, 12998815: 992, 13037406: 993, 13040303: 994, 13044778: 995, 13052670: 996, 13054560: 997, 13133613: 998, 15075141: 999}

img_dir = args.img_dir
class_id = img_dir.split('/')[1].split('_')[0]
target = IMAGENET[int(class_id[1:])]
print('class_id:', class_id)
print('target:', target)

### Random Seed
# torch.manual_seed(1)
# torch.cuda.manual_seed_all(1)
# print('Random Seed OK!')
date = datetime.date.today()
basic_path = args.train_url
'''basic_path = './output/'
basic_path += str(args.job_id) + '_'
if args.name:
    basic_path += str(args.name)
basic_path = basic_path \
             + '_' + str(date.month) + '_' + str(date.day) \
             + '_classid_' + str(class_id) \
             + '_truncation_' + str(args.truncation) \
             + '_lr1_' + str(args.lr1) \
             + '_lr2_' + str(args.lr2) \
             + '_e1_' + str(args.epoch1) \
             + '_e2_' + str(args.epoch2)'''

print('basic_path:', basic_path)

record_file = basic_path + 'train_process.txt'
acc_file = basic_path + 'acc_epoch.txt'
loss_file = basic_path + 'loss_epoch.txt'
# CV_path = basic_path + './CoVariance/'

# Load pre-trained model tokenizer (vocabulary)
netG = BigGAN.from_pretrained('biggan-deep-512')

class counter():
    def __init__(self):
        self.value = 0
        self.ave = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.ave = self.sum / self.count



def topk_accuracy(output, labels, k):
    batch_size = labels.size(0)
    _, pred = output.topk(k, 1, True, True)

    pred = pred.t()
    judge = pred.eq(labels.view(1, -1).expand_as(pred))
    acc_top_k = judge[:k].view(-1).float().sum(0)*100.0/batch_size
    return acc_top_k


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


'''def _image_restore(normalized_image, flag):
    if flag == 'train':
        return normalized_image.mul(mask_train_std[:normalized_image.size(0)]) + mask_train_mean[:normalized_image.size(0)]
    elif flag == 'test':
        return normalized_image.mul(mask_test_std[:normalized_image.size(0)]) + mask_test_mean[:normalized_image.size(0)]
def _image_norm(ini_image, flag):
    if flag == 'train':
        return (ini_image - mask_train_mean[:ini_image.size(0)]).div(mask_train_std[:ini_image.size(0)])
    elif flag == 'test':
        return (ini_image - mask_test_mean[:ini_image.size(0)]).div(mask_test_std[:ini_image.size(0)])'''

def _image_restore(normalized_image):
    return normalized_image.mul(std_mask) + mean_mask

def _image_norm(ini_image):
    return (ini_image-mean_mask).div(std_mask + 1e-7)


def F_inverse(model, netG, input, class_vector, features_ini, feature_objective):
    truncation = args.truncation
    noise_vector = torch.tensor(
        truncated_noise_sample(truncation=args.truncation, batch_size=1, seed=int(args.noise_seed))
        # truncated_noise_sample(truncation=args.truncation, batch_size=1)
    ).cuda()

    noise_vector.requires_grad = True
    print('initial noise_vector:', noise_vector)


    '''Step 1'''
    opt1 = optim.SGD([{'params': noise_vector}], lr=args.lr1, momentum=0.9, weight_decay=1e-4, nesterov=True)
    # opt1 = optim.Adam([{'params': noise_vector}], lr=args.lr1, weight_decay=1e-4)

    for epoch in range(args.epoch1):
        if epoch == 800:
            for paras in opt1.param_groups:
                paras['lr'] /= 10
        if epoch == 2000:
            for paras in opt1.param_groups:
                paras['lr'] /= 10

        if epoch == 3500:
            for paras in opt1.param_groups:
                paras['lr'] /= 10


        noise_vector_normalized = (noise_vector - noise_vector.mean()).div(noise_vector.std())
        fake_img = netG(noise_vector_normalized, class_vector, truncation).mul(0.5).add(0.5)  # 生成图像标准化，（0.5, 1）

        if epoch % 500 == 0:
            save_images.save_images(fake_img.detach().cpu().numpy(), \
                                    '{0}/images/step1_epoch_{1}.jpg'.format(basic_path, epoch//500))

        input = input.cuda()
        input_interpolate = F.interpolate(input, size=(512, 512), mode='bilinear', align_corners=False).cuda()
        fake_img_norm = _image_norm(fake_img).cuda()
        features_ini = features_ini.cuda()

        # 224的图片进行双线性插值，用来reconstruct
        # 512的照片通过
        _, feature_fake_img = model(fake_img_norm[:, :, 256-112:256+112, 256-112:256+112], isda=True)

        # loss1和loss2的权重比应该调整
        loss1 = 0.01 * torch.sum(torch.abs(fake_img_norm - input_interpolate).pow(2))
        loss2 = torch.sum((feature_fake_img - features_ini).pow(2))
        loss_a = loss1 + loss2
        opt1.zero_grad()
        loss_a.backward(retain_graph=True)
        opt1.step()

        if epoch % 10 == 0:
            print('Step1: Epoch: %d  loss_step1_total: %.1f  loss_1: %.5f  loss_2: %.5f' %
                  (epoch, loss_a.data.item(), loss1.data.item(), loss2.data.item()))


    '''Step 2'''
    # 复制Step1的重构noise并保存
    print('====> Save Reconstruct Noise Vector')
    print('noise_vector:\n', noise_vector)

    _tmp = noise_vector.clone()
    tmp = np.array(_tmp.detach().cpu().numpy())
    np.savetxt('{0}/images/class_{1}_noise_vector.csv'.format(basic_path, target), tmp, delimiter=' ')


    for aug_iter in range(feature_objective.size(0)):
        noise_vector_clone = torch.tensor(noise_vector.clone(), requires_grad=True).cuda()
        # noise_vector_clone.requires_grad = True
        print('noise_vector_clone:\n', noise_vector_clone)
        print('feature_objective[{}]:\n'.format(aug_iter), feature_objective[aug_iter])

        opt2 = optim.SGD([{'params': noise_vector_clone}], lr=args.lr2, momentum=0.9, weight_decay=1e-4, nesterov=True)
        # opt2 = optim.Adam([{'params': noise_vector_clone}], lr=args.lr2, weight_decay=1e-4)

        for epoch in range(args.epoch2):
            if epoch == 1200:
                for paras in opt2.param_groups:
                    paras['lr'] /= 10
            if epoch == 2500:
                for paras in opt2.param_groups:
                    paras['lr'] /= 10
            if epoch == 4000:
                for paras in opt2.param_groups:
                    paras['lr'] /= 10

            # noise_vector_normalized = (recon_z - recon_z.mean(0).view(1, 100, 1, 1)
            #                         .expand(args.batch_size, 100, 1, 1)) \
            #                         .div(recon_z.std(0).view(1, 100, 1, 1)
            #                         .expand(args.batch_size, 100, 1, 1))
            #
            noise_vector_normalized = (noise_vector_clone - noise_vector_clone.mean()).div(noise_vector_clone.std())
            fake_img = netG(noise_vector_normalized, class_vector, truncation).mul(0.5).add(0.5)

            if epoch % 500 == 0:
                save_images.save_images(fake_img.detach().cpu().numpy(), \
                                        '{0}/images/step2_augiter_{1}_epoch_{2}.jpg'.format(basic_path, aug_iter, epoch // 500))

            fake_img_norm = _image_norm(fake_img).cuda()
            _, feature_fake_img = model(fake_img_norm[:, :, 256-112:256+112, 256-112:256+112], isda=True)
            loss_b = torch.sum((feature_fake_img - feature_objective[aug_iter].cuda()).pow(2))

            opt2.zero_grad()
            loss_b.backward(retain_graph=True)
            opt2.step()

            if epoch % 10 == 0:
                print('Step2: AugIter: %d  Epoch: %d  loss_step2_total: %.1f'% (aug_iter, epoch, loss_b.data.item()))
                print(noise_vector_clone)
        losses = counter()
        accuracy = counter()
        batch_time = counter()
        log_loss = counter()
        end = time.time()

        fake_img_aug = netG(noise_vector_clone, class_vector, truncation).mul(0.5).add(0.5)
        save_images.save_images(fake_img_aug.detach().cpu().numpy(), \
                                '{0}/images/step2_final.jpg'.format(basic_path))


        loss_re_construct = torch.sum((fake_img_aug - _image_restore(input_interpolate)).pow(2))
        loss_re_construct /= (224 * 224 * 3)
        log_loss.update(loss_re_construct.data.item())

        # optimizer.zero_grad()

        # loss_re_construct.backward()
        # optimizer.step()

        # print statistics
        batch_time.update(time.time() - end)
        end = time.time()

        # print(discriminate_weights)
        fd = open(record_file, 'a+')
        string = ('Aug Iter: [{}]\t'
                  'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                  'Log_Loss {log_loss.value:.4f} '
                  '({log_loss.ave:.4f})\t'
                  'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(
            aug_iter, batch_time=batch_time, top1=accuracy, log_loss=log_loss))

        print(string)
        # print(weights)
        fd.write(string + '\n')
        fd.close()


def train(input, target, netG, model):

    class_vector = torch.zeros(1, 1000)
    class_vector[0][target] = 1

    netG.train()
    model.train()

    model = model.cuda()
    netG = netG.cuda()
    class_vector = class_vector.cuda()

    print('====> Test Image Norm & Restore')
    # print(_image_restore(input).size())
    input_interpolate = F.interpolate(input, size=(512, 512), mode='bilinear', align_corners=False).cuda()
    save_images.save_images(_image_restore(input_interpolate).cpu().detach().numpy(), '{0}/images/raw.jpg'.format(basic_path))
    save_images.save_images(_image_restore(_image_norm(_image_restore(input_interpolate))).cpu().detach().numpy(),
                            '{0}/images/raw_test_restore.jpg'.format(basic_path))

    # inputs = input.expand(args.aug_number, input.size(0), input.size(1), input.size(2)).cuda()
    input = input.cuda()
    print('input:', input.size())
    _, feature_ini = model(input, isda=True)

    # TODO: 根据cov增广得到feature_objective
    print('====> Preparing Covariance')
    var_dir = './Covariance/' + str(target) + '_cov_imagenet' + '.csv'
    print(var_dir)

    var = np.loadtxt(var_dir, delimiter=' ')
    CV = np.diag(var)
    print('CV:', CV.shape)

    # augmentation高斯采样
    feature_objective = torch.zeros(args.aug_number, 2048)
    for i in range(args.aug_number):
        aug_noise = np.random.multivariate_normal([0 for j in range(var.shape[0])], args.aug_alpha * CV)
        aug_noise = torch.Tensor(aug_noise).cuda()
        feature_objective[i] = feature_ini + aug_noise   # feature_objective: \tilde{a_i}

    print('feature_objective:', feature_objective.size())


    print('====> Start Training')
    # z_trained = F_inverse(model, netG, input, class_vector, feature_ini, feature_objective)
    F_inverse(model, netG, input, class_vector, feature_ini, feature_objective)

    '''fake_img_aug = netG(z_trained, class_vector).mul(0.5).add(0.5)
    save_images.save_images(fake_img_aug.detach().cpu().numpy(), \
                            '{0}/step2_{1}_fake.jpg'.format(basic_path, i))'''




def validate(testloader, model_pre, decoder, epoch, acc_epoch, loss_epoch):
    losses = counter()
    accuracy = counter()
    batch_time = counter()
    train_batches_num = len(testloader)
    end = time.time()

    # Amount = torch.zeros(class_num).cuda()

    model_pre.eval()
    decoder.eval()
    for i, data in enumerate(testloader, 0):

        # get the inputs
        inputs, labels = data

        inputs, labels = inputs.cuda(), labels.cuda(async=True)

        features = model_pre(inputs)

        re_construct = decoder(features.view(inputs.size(0), feature_num, 1, 1).detach())

        if i == 0:
            save_images.save_images(re_construct.detach().cpu().numpy(),
                                    './Class_condition' + str(args.class_id) +'/cifar10/test_reconstruct_epoch_{}.jpg'.format(epoch))
            save_images.save_images(_image_restore(inputs, 'test').cpu().numpy(),
                                    './Class_condition' + str(args.class_id) +'/cifar10/test_initial_epoch_{}.jpg'.format(epoch))


        loss_re_construct = torch.sum((re_construct - _image_restore(inputs, 'train')).pow(2))
        loss_re_construct /= (labels.size(0) * 224 * 224 * 3)
        losses.update(loss_re_construct.data.item(), labels.size(0))


        # optimizer.zero_grad()

        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0:
            fd = open(record_file, 'a+')
            string = ('Test: [{0}][{1}/{2}]\t'
                      'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                      'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
                      'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(
                       epoch, (i+1), train_batches_num, batch_time=batch_time,
                       loss=losses, top1=accuracy))
            print(string)
            fd.write(string + '\n')
            fd.close()

    fd = open(record_file, 'a+')
    string = ('Test: [{0}][{1}/{2}]\t'
              'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
              'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
              'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(
        epoch, train_batches_num, train_batches_num, batch_time=batch_time,
        loss=losses, top1=accuracy))
    print(string)
    fd.write(string + '\n')
    fd.close()


    acc_epoch.append(accuracy.ave)
    loss_epoch.append(losses.ave)



def main():
    acc_epoch = []
    loss_epoch = []

    if args.cloud:
        isExists = os.path.exists(basic_path)
        if not isExists:
            os.makedirs(basic_path)


    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    trans = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    dir = args.train_url + args.img_dir
    raw_img = Image.open(dir)
    input = trans(raw_img)
    input = input.view(1, input.size(0), input.size(1), input.size(2))
    print('input:', input.size())



    ### Models
    print('=========> Load Checkpoint from train_url dir')
    checkpoint_dir = os.path.join(args.train_url, 'resnet50-19c8e357.pth')
    print('checkpoint_dir:', checkpoint_dir)
    checkpoint = torch.load(checkpoint_dir)
    model = networks_imagenet.resnet.resnet50()
    model.load_state_dict(checkpoint)
    netG = BigGAN.from_pretrained('biggan-deep-512', model_dir=args.train_url)


    # _, feature = model.forward(input, isda=True)
    # print(feature.size())

    train(input, target, netG, model)

    # model.load_state_dict(torch.load('./model_110_PRE'))
    # netG.load_state_dict(torch.load('./WGAN-Generators/Results/Model' + str(args.class_id) + '/generator.pkl'))
    # netG.load_state_dict(torch.load('./generator_all.pkl'))
    # optimizer = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

    # validate(testloader, model, netG, epoch, acc_epoch, loss_epoch)


if __name__ == '__main__':
    main()