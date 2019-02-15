from tn.network import TweetAnalysisNetwork
import argparse


print('\n|-----------------------------'
        '---------------------------------------|')
print('|     TRAINING NEURAL NETWORK FOR TWEEET SENTIMENT ANALYSIS          |')
print('|----------------------------'
        '----------------------------------------|', end='\n\n')

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--destination', dest="destination",
        help="Path to the scv file with dataset")

parser.add_argument('-s', '--save', dest="save",
        help="Path to the folder where model will be saved")

parser.add_argument('-n', '--name',
        help="Name of the model which will be save")

args = parser.parse_args()

destination = args.destination
save = args.save
name = args.name

errors = []

if not destination:
    errors.append('\t*Provide path. To the folder where csv dataset could'
            'be found. \n\t You can pass is'
            'by \'-d\' or \'--destination\' flag')

if not save:
    errors.append('\t*Provide path. Path to the folder where'
        ' model will be saved.\n\t You can pass is by'
        ' \'-s\' or \'--save\' flag')

if not name:
    errors.append('\t*Provide model name.'
            ' You can pass it by \'-n\' or \'--name\' flag')

if len(errors) > 0:
    print('  ## ERRORS')
    for error in errors:
        print(error)
else:
    network = TweetAnalysisNetwork(name=name, data_path=destination)
    network.create()
    network.compile()
    network.train()
    network.save(save)

print('\n|----------------------------------'
        '----------------------------------|')
print('|--------------------------------------'
        '------------------------------|', end='\n\n')
