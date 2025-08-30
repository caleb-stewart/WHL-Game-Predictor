
import sys
import asyncio

if sys.platform.startswith("win"):
# Set the correct event loop policy for Windows. IDK if this does anything... the warning is still there
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


import os
import json

notebook_name = "WHL_Win_Predictor.ipynb"

with open(notebook_name, 'r', encoding='utf-8') as notebook_file:
    notebook_content = json.load(notebook_file)

# ensure we are using the right kernel to be ran in the command line
notebook_content['metadata']['kernelspec']['display_name'] = 'Python 3'
notebook_content['metadata']['kernelspec']['language'] = 'python'
notebook_content['metadata']['kernelspec']['name'] = 'python3'
print('Rewrote kernel spec to: python3')

# rewrite the notebook file with the correct kernel
with open(notebook_name, 'w') as notebook_file:
    json.dump(notebook_content, notebook_file, indent=4)

print('executing notebook: ', notebook_name)
# Execute the notebook in-place
os.system(f'jupyter nbconvert --to notebook --execute {notebook_name}')