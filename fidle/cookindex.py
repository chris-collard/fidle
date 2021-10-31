
# -----------------------------------------------------------------------------
#                ____                 _      ___           _
#              / ___|___   ___   ___| | __ |_ _|_ __   __| | _____  __
#             | |   / _ \ / _ \ / __| |/ /  | || '_ \ / _` |/ _ \ \/ /
#             | |__| (_) | (_) | (__|   <   | || | | | (_| |  __/>  <
#             \____\___/ \___/ \___|_|\_\ |___|_| |_|\__,_|\___/_/\_\
#
#                                                   Fidle mod for index cooking
# -----------------------------------------------------------------------------
#
# A simple module to build the notebook catalog and update the README.
# Jean-Luc Parouty 2021


import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import pandas as pd
from IPython.display import display, Markdown, HTML

import re
import sys, os, glob, yaml
import json
from datetime import datetime
from collections import OrderedDict
from IPython.display import display

sys.path.append('..')
import fidle.config as config

# -----------------------------------------------------------------------------
# To built README.md / README.ipynb
# -----------------------------------------------------------------------------
#    get_files          :  Get files lists
#    get_notebook_infos :  Get infos about a entry
#    get_catalog        :  Get a catalog of all entries
# -----------------------------------------------------------------------------

def build_catalog(directories):

    # ---- Get the notebook list
    #
    files_list = get_files(directories.keys())

    # ---- Get a detailled catalog for this list
    #
    catalog = get_catalog(files_list)

    with open(config.CATALOG_FILE,'wt') as fp:
        n=len(catalog)
        json.dump(catalog,fp,indent=4)
        print(f'Catalog saved as         : {config.CATALOG_FILE} ({n} entries)')


def get_files(directories, top_dir='..'):
    '''
    Return a list of files from a given list of directories
    args:
        directories : list of directories
        top_dir : location of theses directories
    return:
        files : filenames list (without top_dir prefix)
    '''
    files = []
    regex = re.compile('.*==.+?==.*')

    for d in directories:
        notebooks = glob.glob( f'{top_dir}/{d}/*.ipynb')
        notebooks.sort()
        scripts   = glob.glob( f'{top_dir}/{d}/*.sh')
        scripts.sort()
        files.extend(notebooks)
        files.extend(scripts)
        
    files = [x for x in files if not regex.match(x)]
    files = [ x.replace(f'{top_dir}/','') for x in files]
    return files


def get_notebook_infos(filename, top_dir='..'):
    '''
    Extract informations from a fidle notebook.
    Informations are dirname, basename, id, title, description and are extracted from comments tags in markdown.
    args:
        filename : notebook filename
    return:
        dict : with infos.
    '''
    # print('Read : ',filename)
    about={}
    about['id']          = '??'
    about['dirname']     = os.path.dirname(filename)
    about['basename']    = os.path.basename(filename)
    about['title']       = '??'
    about['description'] = '??'
    about['overrides']   = None
    
    # ---- Read notebook
    #
    notebook = nbformat.read(f'{top_dir}/{filename}', nbformat.NO_CONVERT)
    
    # ---- Get id, title and desc tags
    #
    overrides=[]
    for cell in notebook.cells:
     
        # ---- Find Index informations
        #
        if cell['cell_type'] == 'markdown':

            find = re.findall(r'<\!-- TITLE -->\s*\[(.*)\]\s*-\s*(.*)\n',cell.source)
            if find:
                about['id']    = find[0][0]
                about['title'] = find[0][1]

            find = re.findall(r'<\!-- DESC -->\s*(.*)\n',cell.source)
            if find:
                about['description']  = find[0]

        # ---- Find override informations
        #
        if cell['cell_type'] == 'code':
            
            # Try to find : override(...) call
            for m in re.finditer('override\((.+?)\)', cell.source):
                overrides.extend ( re.findall(r'\w+', m.group(1)) )

            # Try to find : run_dir=
            if re.search(r"\s*run_dir\s*?=", cell.source):
                overrides.append('run_dir')
                
    about['overrides']=overrides
    return about

    
    
def get_txtfile_infos(filename, top_dir='..'):
    '''
    Extract fidle  informations from a text file (script...).
    Informations are dirname, basename, id, title, description and are extracted from comments tags in document
    args:
        filename : file to analyze
    return:
        dict : with infos.
    '''

    about={}
    about['id']          = '??'
    about['dirname']     = os.path.dirname(filename)
    about['basename']    = os.path.basename(filename)
    about['title']       = '??'
    about['description'] = '??'
    about['overrides']   = []
    
    # ---- Read file
    #
    with open(f'{top_dir}/{filename}') as fp:
        text = fp.read()

    find = re.findall(r'<\!-- TITLE -->\s*\[(.*)\]\s*-\s*(.*)\n',text)
    if find:
        about['id']    = find[0][0]
        about['title'] = find[0][1]

    find = re.findall(r'<\!-- DESC -->\s*(.*)\n',text)
    if find:
        about['description']  = find[0]

    return about

              
def get_catalog(files_list=None, top_dir='..'):
    '''
    Return an OrderedDict of files attributes.
    Keys are file id.
    args:
        files_list : list of files filenames
        top_dir : Location of theses files
    return:
        OrderedDict : {<file id> : { description} }
    '''
       
    catalog = OrderedDict()

    # ---- Build catalog
    for file in files_list:
        about=None
        if file.endswith('.ipynb'): about = get_notebook_infos(file, top_dir='..')
        if file.endswith('.sh'):    about = get_txtfile_infos(file, top_dir='..')
        if about is None:
            print(f'** Warning : File [{file}] have no tags infomations...')
            continue
        id=about['id']
        catalog[id] = about
        
    return catalog
        

def tag(tag, text, document):
    '''
    Put a text inside a tag
    args:
        tag : tag prefix name
        txt : text to insert
        document : document 
    return:
        updated document
    '''
    debut  = f'<!-- {tag}_BEGIN -->'
    fin    = f'<!-- {tag}_END -->'

    output = re.sub(f'{debut}.*{fin}',f'{debut}\n{text}\n{fin}',document, flags=re.DOTALL)
    return output


def read_catalog():
    '''
    Read json catalog file.
    args:
        None
    return:
        json catalog
    '''
    with open(config.CATALOG_FILE) as fp:
        catalog = json.load(fp)
    return catalog


# -----------------------------------------------------------------------------
# To built default.yml profile
# -----------------------------------------------------------------------------
#    build_default_profile :  Get default profile
# -----------------------------------------------------------------------------


def build_default_profile(output_tag='==ci=='):
    '''
    Return a default profile for continous integration.
    Ce profile contient une liste des notebooks avec les paramètres modifiables.
    Il peut être modifié et sauvegardé, puis être utilisé pour lancer l'éxécution
    des notebooks.
    params:
        catalog : Notebooks catalog. if None (default), load config.CATALOG_FILE
        output_tag  : tag name of generated notebook
        profile_filename : Default profile filename
    return:
        None
    '''
    
    catalog = read_catalog()

    metadata   = { 'version'       : '1.0', 
                   'output_tag'    : output_tag, 
                   'save_figs'     : True, 
                   'description'   : 'Default generated profile',
                   'output_ipynb'  : '<directory for ipynb>',
                   'output_html'   : '<directory for html>',
                   'report_json'   : '<report json file>',
                   'report_error'  : '<error file>'
                   }
    profile  = { '_metadata_':metadata }
    for id, about in catalog.items():
        
        id        = about['id']
        title     = about['title']
        dirname   = about['dirname']
        basename  = about['basename']
        overrides = about.get('overrides',None)
    
        notebook = {}
        notebook['notebook_id']  = id
        notebook['notebook_dir'] = dirname
        notebook['notebook_src'] = basename
        notebook['notebook_tag'] = 'default'
        if len(overrides)>0:
            notebook['overrides']={ name:'default' for name in overrides }
                    
        profile[f'Nb_{id}']=notebook
        
    # ---- Save profile
    #
    with open(config.PROFILE_FILE,'wt') as fp:
        n=len(profile)-1
        yaml.dump(profile, fp, sort_keys=False)
        print(f'default profile saved as : {config.PROFILE_FILE} ({n} entries)')