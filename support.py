import os

#rename file in forlder test_file/calculate
for idx, filename in enumerate(os.listdir('test_file/tuyen')):
    os.rename('test_file/tuyen/'+filename, 'test_file/tuyen/'+'tuyen_' + str(idx) + '.jpg')