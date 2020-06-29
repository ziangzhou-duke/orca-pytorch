output_files = ['train', 'test', 'devel']
file = 'whole_utt2wav'
for ofile in output_files:
    input_file = open(file, "rt")
    output_file = open("../tmp/" + ofile + '_utt2wav', "wt")
    
    for line in input_file:
        if ofile in line:
            output_file.write(line.replace('/data1/wuhw/compare/data/', '/home/jovyan/work/hpc4/orca_tpl/wav/'))

    input_file.close()
    output_file.close()