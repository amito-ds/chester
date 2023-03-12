import json

from jinja2 import Template

# Load the template file
with open('/Users/amitosi/PycharmProjects/chester/example_notebooks/template.ipynb') as f:
    template_str = f.read()

# Define the parameters to change
names = ['cpu_small',
         'spoken-arabic-digit',
         'new3s.wc',
         'QSAR-TID-191',
         'letter',
         'w1a',
         'QSAR-TID-11969',
         'segment',
         'mtp',
         'eye_movements',
         'mnist_784',
         'OVA_Lung',
         'QSAR-TID-226',
         'musk',
         'one-hundred-plants-shape',
         'connect-4',
         'one-hundred-plants-margin',
         'pc1',
         'volcanoes-b3',
         'cardiotocography',
         'QSAR-TID-11755',
         'QSAR-TID-11242',
         'ringnorm',
         'steel-plates-fault',
         'QSAR-TID-11',
         'w2a',
         'coil2000',
         'OVA_Breast',
         'Click_prediction_small',
         'QSAR-TID-12687',
         'svmguide1',
         'cpu_small',
         'la2s.wc',
         'vehicle_sensIT',
         'mfeat-zernike',
         'eeg-eye-state',
         'nursery',
         'real-sim',
         'QSAR-TID-11017',
         'quake',
         'QSAR-TID-11631',
         'kin8nm',
         'QSAR-TID-100080',
         'QSAR-TID-250',
         'pendigits',
         'wind',
         'delta_ailerons',
         'analcatdata_halloffame',
         'abalone',
         'w8a',
         'page-blocks',
         'cardiotocography',
         'mv',
         'isolet',
         'cmc',
         'poker',
         'QSAR-TID-10979',
         'QSAR-TID-65',
         'bank32nh',
         'satellite_image',
         'QSAR-TID-10280',
         'volcanoes-d1',
         'baseball',
         'jm1',
         'Click_prediction_small',
         'kr-vs-k',
         'nomao',
         'volcanoes-e1',
         'car',
         'ldpa',
         'SensIT-Vehicle-Combined',
         'spambase',
         'quake',
         'space_ga',
         'kr-vs-kp',
         'CovPokElec',
         'mfeat-morphological',
         'Stagger3',
         'artificial-characters',
         'QSAR-TID-100044']
import os

# Render the template for each parameter value
for name in names:
    # Create a new template with the parameter value
    template = Template(template_str)
    rendered_template = template.render(name=name)

    # Load the rendered template as a JSON object
    nb = json.loads(rendered_template)

    # Replace the "db_name" with the current name
    for cell in nb['cells']:
        if 'db_name' in ''.join(cell.get('source', '')):
            print("original ", cell['source'][0])
            cell['source'][0] = cell['source'][0].replace('db_name', f"'{name}'")
            print("replaced", cell['source'][0])

    # Save the modified notebook to a new file in the "notebooks" subfolder
    if not os.path.exists('notebooks'):
        os.makedirs('notebooks')
    with open(os.path.join('notebooks', f'{name}.ipynb'), 'w') as f:
        json.dump(nb, f)
