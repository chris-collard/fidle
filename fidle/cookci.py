
# -----------------------------------------------------------------------------
#                         ____                 _       ____ _
#                       / ___|___   ___   ___| | __  / ___(_)
#                      | |   / _ \ / _ \ / __| |/ / | |   | |
#                      | |__| (_) | (_) | (__|   <  | |___| |
#                      \____\___/ \___/ \___|_|\_\  \____|_|
#
#                                           Fidle mod for continous integration
# -----------------------------------------------------------------------------
#
# A simple module to run all notebooks with parameters overriding
# Jean-Luc Parouty 2021

import sys,os,platform
import json
import datetime, time
import nbformat
from nbconvert               import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError
from asyncio import CancelledError
import re
import yaml
import base64
from collections import OrderedDict
from IPython.display import display,Image,Markdown,HTML
import pandas as pd

sys.path.append('..')
import fidle.config as config
import fidle.cookindex as cookindex

VERSION = '1.4.1'

start_time = {}
end_time   = {}

_report_json  = None
_report_error = None


       
def load_profile(filename):
    '''Load yaml profile'''
    with open(filename,'r') as fp:
        profile=yaml.load(fp, Loader=yaml.FullLoader)
        print(f'\nLoad profile :{filename}')
        print('  - Entries : ',len(profile)-1)
        return profile
    
    
def run_profile(profile_name, reset=False, filter=r'.*', top_dir='..'):
    '''
    Récupère la liste des notebooks et des paramètres associés,
    décrit dans le profile, et pour chaque notebook :
    Positionner les variables d'environnement pour l'override
    Charge le notebook
    Exécute celui-ci
    Sauvegarde le notebook résultat, avec son nom taggé.
    Params:
        profile_name : nom du profile d'éxécution
        top_dir : chemin relatif vers la racine fidle (..)
    '''

    print(f'\n=== Run profile session - FIDLE 2021')
    print(f'=== Version : {VERSION}')
    
    chrono_start('main')
    
    # ---- Retrieve profile ---------------------------------------------------
    #
    profile   = load_profile(profile_name)
    config    = profile['_metadata_']
    notebooks = profile
    del notebooks['_metadata_']   
    
    # ---- Report file --------------------------------------------------------
    #
    metadata = config
    metadata['host']    = platform.uname()[1]
    metadata['profile'] = profile_name

    report_json  = top_dir + '/' + config['report_json' ]
    report_error = top_dir + '/' + config['report_error']

    init_ci_report(report_json, report_error, metadata, reset=reset)
    
    # ---- Where I am, me and the output_dir ----------------------------------
    #
    home         = os.getcwd()
    output_ipynb = config['output_ipynb']
    output_html  = config['output_html']
        
    # ---- Environment vars ---------------------------------------------------
    #
    print('\nSet environment var:')
    environment_vars = config['environment_vars']
    for name,value in environment_vars.items():
        os.environ[name] = str(value)
        print(f'  - {name:20s} = {value}')

    # ---- For each notebook --------------------------------------------------
    #
    print('\n---- Start running process ------------------------')
    for run_id,about in notebooks.items():

        # ---- Filtering ------------------------------------------------------
        #
        if not re.match(filter, run_id):
            continue
        
        print(f'\n  - Run : {run_id}')

        # ---- Get notebook infos ---------------------------------------------
        #
        notebook_id   = about['notebook_id']
        notebook_dir  = about['notebook_dir']
        notebook_src  = about['notebook_src']
        notebook_name = os.path.splitext(notebook_src)[0]
        notebook_tag  = about['notebook_tag']
        overrides     = about.get('overrides',None)
        

        # ---- Output name ----------------------------------------------------
        #
        if notebook_tag=='default':
            output_name  = notebook_name + config['output_tag']
        else:
            output_name  = notebook_name + notebook_tag
 
        # ---- Go to the right place ------------------------------------------
        #
        os.chdir(f'{top_dir}/{notebook_dir}')

        # ---- Override ------------------------------------------------------- 
        
        to_unset=[]
        if isinstance(overrides,dict):
            print('    - Overrides :')
            for name,value in overrides.items():
                # ---- Default : no nothing
                if value=='default' : continue
                #  ---- Set env
                env_name  = f'FIDLE_OVERRIDE_{notebook_id.upper()}_{name}'
                env_value = str(value)
                os.environ[env_name] = env_value
                # ---- For cleaning
                to_unset.append(env_name)
                # ---- Fine :(-)
                print(f'      - {env_name:38s} = {env_value}')
    
        # ---- Run it ! -------------------------------------------------------

        # ---- Go to the notebook_dir
        #
        os.chdir(f'{top_dir}/{notebook_dir}')

        # ---- Read notebook
        #
        notebook = nbformat.read( f'{notebook_src}', nbformat.NO_CONVERT)

        # ---- Top chrono - Start
        #
        chrono_start('nb')
        update_ci_report(run_id, notebook_id, notebook_dir, notebook_src, output_name, start=True)
        
        # ---- Try to run...
        #
        print('    - Run notebook...',end='')
        try:
            ep = ExecutePreprocessor(timeout=6000, kernel_name="python3")
            ep.preprocess(notebook)
        except CellExecutionError as e:
            happy_end = False
            output_name = notebook_name + '==ERROR=='
            print('\n   ','*'*60)
            print( '    ** AAARG.. An error occured : ',type(e).__name__)
            print(f'    ** See notebook :  {output_name} for details.')
            print('   ','*'*60)
        else:
            happy_end = True
            print('..done.')

        # ---- Top chrono - Stop
        #
        chrono_stop('nb')        
        update_ci_report(run_id, notebook_id, notebook_dir, notebook_src, output_name, end=True, happy_end=happy_end)
        print('    - Duration : ',chrono_get_delay('nb') )

        # ---- Back to home
        #
        os.chdir(home)

        # ---- Check for images to embed --------------------------------------
        #      We just try to embed <img src="..."> tag in some markdown cells
        #      Very fast and suffisant for header/ender.
        #
        for cell in notebook.cells:
            if cell['cell_type'] == 'markdown':
                cell.source = images_embedder(cell.source)

        # ---- Save notebook as ipynb -----------------------------------------
        #
        if output_ipynb.lower()!="none":
            save_dir = os.path.abspath( f'{top_dir}/{output_ipynb}/{notebook_dir}' )
            os.makedirs(save_dir, mode=0o750, exist_ok=True)
            with open(  f'{save_dir}/{output_name}.ipynb', mode="w", encoding='utf-8') as fp:
                nbformat.write(notebook, fp)
            print(f'    - Saved {save_dir}/{output_name}.ipynb')

        # ---- Save notebook as html ------------------------------------------
        #
        if output_html.lower()!="none":

            # ---- Convert notebook to html
            exporter = HTMLExporter()
            exporter.template_name = 'classic'
            (body_html, resources_html) = exporter.from_notebook_node(notebook)
            
            # ---- Embed images
            # body_html = images_embedder(body_html)
            
            # ---- Save
            save_dir = os.path.abspath( f'{top_dir}/{output_html}/{notebook_dir}' )
            os.makedirs(save_dir, mode=0o750, exist_ok=True)
            with open(  f'{save_dir}/{output_name}.html', mode='wb') as fp:
                fp.write(body_html.encode("utf-8"))
            print(f'    - Saved {save_dir}/{output_name}.html')

        # ---- Clean all ------------------------------------------------------
        #
        for env_name in to_unset:
            del os.environ[env_name]

    # ---- End of running
    chrono_stop('main')
    print('\n---- End of running process -----------------------')

    print('\nDuration :', chrono_get_delay('main'))
    complete_ci_report()
    
    

def get_base64_image(filename):
    '''
    Read an image file and return as a base64 encoded version
    params:
        filename : image filemane
    return:
        base 64 encoded image
    '''
    with open(filename, "rb") as image_file:
        img64 = base64.b64encode(image_file.read())
    src="data:image/svg+xml;base64,"+img64.decode("utf-8") 
    return src

    
def images_embedder(html):
    '''
    Images embedder. Search images src="..." link and replace them
    by base64 embedded images.
    params:
        html: input html
    return:
        output html
    '''
    for img_tag in re.findall('.*(<img .*>).*', html):
        for src_tag in re.findall('.*src=[\'\"](.*)[\'\"].*', img_tag):
            if src_tag.startswith('data:'): continue
            src_b64 = get_base64_image(src_tag)
            img_b64 = img_tag.replace(src_tag, src_b64)
            html = html.replace(img_tag,img_b64)
    return html




def chrono_start(id='default'):
    global start_time
    start_time[id] = datetime.datetime.now()
        
def chrono_stop(id='default'):
    global end_time
    end_time[id] = datetime.datetime.now()

def chrono_get_delay(id='default', in_seconds=False):
    global start_time, end_time
    delta = end_time[id] - start_time[id]
    if in_seconds:
        return round(delta.total_seconds(),2)
    else:
        delta = delta - datetime.timedelta(microseconds=delta.microseconds)
        return str(delta)

def chrono_get_start(id='default'):
    global start_time
    return start_time[id].strftime("%d/%m/%y %H:%M:%S")

def chrono_get_end(id='default'):
    global end_time
    return end_time[id].strftime("%d/%m/%y %H:%M:%S")

def reset_chrono():
    global start_time, end_time
    start_time, end_time = {},{}
    

def init_ci_report(report_json, report_error, metadata, reset=True):
    
    global _report_json, _report_error
    
    _report_json  = os.path.abspath(report_json)
    _report_error = os.path.abspath(report_error)

    print('\nInit report :')
    print(f'  - report file is : {_report_json}')
    print(f'  - error  file is : {_report_error}')

    # ---- Create directories if doesn't exist
    #
    report_dir=os.path.dirname(report_json)
    os.makedirs(report_dir, mode=0o750, exist_ok=True)
    
    # ---- Reset ?
    #
    if reset is False and os.path.isfile(_report_json) :
        with open(_report_json) as fp:
            report = json.load(fp)
        print('  - Report is reloaded')
    else:
        report={}
        print('- Report is reseted')

    metadata['reseted']=reset     
    metadata['start']=chrono_get_start('main')

    # ---- Create json report
    #
    report['_metadata_']=metadata
    with open(_report_json,'wt') as fp:
        json.dump(report,fp,indent=4)
    print('  - Report file saved')

    # ---- Reset error
    #
    if os.path.exists(_report_error):
        os.remove(_report_error)
    print('  - Error file removed')

    
def complete_ci_report():

    global _report_json

    with open(_report_json) as fp:
        report = json.load(fp)
        
    report['_metadata_']['end']      = chrono_get_end('main')
    report['_metadata_']['duration'] = chrono_get_delay('main')
    
    with open(_report_json,'wt') as fp:
        json.dump(report,fp,indent=4)
        
    print(f'\nComplete ci report :')
    print(f'  - Report file saved')
    
    
def update_ci_report(run_id, notebook_id, notebook_dir, notebook_src, notebook_out, start=False, end=False, happy_end=True):
    global start_time, end_time
    global _report_json, _report_error
    
    # ---- Load it
    with open(_report_json) as fp:
        report = json.load(fp)
        
    # ---- Update as a start
    if start is True:
        report[run_id]              = {}
        report[run_id]['id']        = notebook_id
        report[run_id]['dir']       = notebook_dir
        report[run_id]['src']       = notebook_src
        report[run_id]['out']       = notebook_out
        report[run_id]['start']     = chrono_get_start('nb')
        report[run_id]['end']       = ''
        report[run_id]['duration']  = 'Unfinished...'
        report[run_id]['state']     = 'Unfinished...'
        report['_metadata_']['end']      = 'Unfinished...'
        report['_metadata_']['duration'] = 'Unfinished...'


    # ---- Update as an end
    if end is True:
        report[run_id]['end']       = chrono_get_end('nb')
        report[run_id]['duration']  = chrono_get_delay('nb')
        report[run_id]['state']     = 'ok' if happy_end else 'ERROR'
        report[run_id]['out']       = notebook_out     # changeg in case of error

    # ---- Save report
    with open(_report_json,'wt') as fp:
        json.dump(report,fp,indent=4)

    if not happy_end:
        with open(_report_error, 'a') as fp:
            print(f"See : {notebook_dir}/{notebook_out} ", file=fp)
        
        


def build_ci_report(profile_name, top_dir='..'):
    
    print('\n== Build CI Report - FIDLE 2021')
    print(f'== Version : {VERSION}')


    profile   = load_profile(profile_name)
    config    = profile['_metadata_']

    report_json  = top_dir + '/' + config['report_json' ]
    report_error = top_dir + '/' + config['report_error']

    report_json  = os.path.abspath(report_json)
    report_error = os.path.abspath(report_error)

    # ---- Load report
    #
    print('\nReport in progress:')
    with open(report_json) as infile:
        report = json.load( infile )
    print(f'  - Load json report file : {_report_json}')

    # ---- metadata
    #
    metadata=report['_metadata_']
    del report['_metadata_']

    output_html = metadata['output_html']

    if output_html.lower()=='none':
        print('  - No HTML output is specified in profile...')
        return
    
    reportfile = os.path.abspath( f'{top_dir}/{output_html}/index.html' )

    # ---- HTML for metadata
    #
    html_metadata = ''
    for name,value in metadata.items():
        html_metadata += f'<b>{name.title()}</b> : {value}  <br>\n'

    # ---- HTML for report    
    #
    html_report = '<table>'
    html_report += '<tr><th>Directory</th><th>Id</th><th>Notebook</th><th>Start</th><th>Duration</th><th>State</th></tr>\n'
    for id,entry in report.items():
        dir   = entry['dir']
        src   = entry['src']
        out   = entry['out']+'.html'
        start = entry['start']
        end   = entry['end']
        dur   = entry['duration']
        state = entry['state']

        cols = []
        cols.append( f'<a href="{dir}"       target="_blank">{dir}</a>'       )
        cols.append( f'<a href="{dir}/{out}" target="_blank">{id}</a>'  )
        cols.append( f'<a href="{dir}/{out}" target="_blank">{src}</a>' )
        cols.append( start )
        cols.append( dur   )
        cols.append( state )

        html_report+='<tr>'
        for c in cols:
            html_report+=f'<td>{c}</td>'
        html_report+='</tr>\n'

    html_report+='</table>'

    body_html = _get_html_report(html_metadata, html_report)
    with open(reportfile, "wt") as fp:
        fp.write(body_html)
    print(f'  - Saved HTML report : {reportfile}')
            

    


def _get_html_report(html_metadata, html_report):

    with open('./img/00-Fidle-header-01.svg','r') as fp:
        logo_header = fp.read()

    with open('./img/00-Fidle-logo-01-80px.svg','r') as fp:
        logo_ender = fp.read()

    html = f"""\
    <html>
        <head><title>FIDLE - CI Report</title></head>
        <body>
        <style>
            body{{
                  font-family: sans-serif;
            }}
            div.title{{ 
                font-size: 1.4em;
                font-weight: bold;
                padding: 40px 0px 10px 0px; }}
            a{{
                color: SteelBlue;
                text-decoration:none;
            }}
            table{{      
                  border-collapse : collapse;
                  font-size : 0.9em;
            }}
            td{{
                  border-style: solid;
                  border-width:  thin;
                  border-color:  lightgrey;
                  padding: 5px 20px 5px 20px;
            }}
            .metadata{{ padding: 10px 0px 10px 30px; font-size: 0.9em; }}
            .result{{ padding: 10px 0px 10px 30px; }}
        </style>

            {logo_header}

            <div class='title'>Notebooks performed :</div>
            <div class="result">
                <p>Here is a "correction" of all the notebooks.</p>
                <p>These notebooks have been run on Jean-Zay, on GPU (V100) and the results are proposed here in HTML format.</p>    
                {html_report}
            </div>
            <div class='title'>Metadata :</div>
            <div class="metadata">
                {html_metadata}
            </div>

            {logo_ender}

            </body>
    </html>
    """
    return html
