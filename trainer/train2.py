from OCHumanApi.ochumanApi.ochuman import OCHuman

ochuman = OCHuman(AnnoFile='../data/ochuman/ochuman.json', Filter='kpt&segm')
image_ids = ochuman.getImgIds()
print ('Total images: %d'%len(image_ids))