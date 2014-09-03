from bic1306 import *

from pyannote.feature.yaafe import YaafeMFCC
mfcc_extractor = YaafeMFCC(e=True, coefs=12)
tolerance = 10
duration = [20] #[10,20,30,40,50]
stpe = 10


Set = 2#[1,2,3,4,5]


Extractor = YaafeMFCC(e=True, coefs=12)
ExpDataSTG = '/people/doumbia/Bureau/global.txt'
global_graph(Dduration, step, tolerance, Set, ExpData, Extractor)
#stg(Seuil_Temp, Seuil_Acoust, Tolerance, EpisodesSet, ExpDataSTG, duration, threshold)