"""Recalculate the protocol-informed proposed assay budget; not measured costs."""
import argparse,json,math,csv
from pathlib import Path
from minicpg.analysis_io import sha256
def calculate(n, **overrides):
 q=p|overrides; pools=q['pools'];reps=q['replicates'];over=1+q['reagent_overage']
 controls=len(q['methylation_levels'])+q['blank_controls']
 reads=q['targets']*q['target_depth']/q['usable_fraction']/q['weakest_to_mean']
 cap=min(q['distinct_barcodes'],math.floor(q['raw_pairs_per_run']/reads))
 assert cap>controls
 runs=math.ceil(n/(cap-controls)); libs=n+runs*controls
 perrun=[n//runs+(i<n%runs)+controls for i in range(runs)]
 assert all(k<=cap and k*reads<=q['raw_pairs_per_run'] for k in perrun)
 rxns=pools*reps*libs;qubits=(pools+2)*libs+8+3*runs;qpcr=27*runs
 plates=math.ceil(rxns/96)+2*math.ceil(pools*libs/96)+4*math.ceil(libs/96)+math.ceil(qpcr/96)
 # Hands-on allowances; exclude outsourced conversion, TapeStation and run operation.
 labor_tasks={'setup_and_control_mixing':1,'target_PCR_setup':rxns/240,'pool_cleanup_and_normalisation':pools*libs/100,'library_setup_and_cleanups':libs/32,'fluorometry_and_pool_QC':libs/80,'pool_assembly_and_records':.5*runs}
 hours=sum(labor_tasks.values())*q.get('labor_multiplier',1)
 tips_per_lib=2*pools*reps+8*pools+10+16+2*(pools+2)+4
 tip_count=math.ceil((tips_per_lib*libs+2*qpcr+20)*q.get('tip_multiplier',1))
 total_beads=(pools*reps*q['pcr_ul']*q['target_bead_ratio']+q['ligation_beads_ul']+q['library_pcr_beads_ul'])*libs
 # Each of the60 target primers appears in three reactions, independent of pool count.
 primer_pmol=2*q['targets']*reps*q['primer_um']*q['pcr_ul']*libs
 rows=[]
 def add(name,qty,unit,refs,basis):rows.append({'component':name,'quantity':qty,'unit_cost':unit,'total':qty*unit,'sources':refs,'basis':basis})
 add('Bisulfite conversion',libs,prices['conversion'],[14],'Published service unit price; one conversion per sample/control, including budgeted blank.')
 add('Target PCR enzyme and buffer',rxns*q['enzyme_u_per_rxn']*over,prices['enzyme'],[2,3],'Enzyme units; manufacturer25µL recipe adapted to multiplex PCR.')
 add('Target PCR dNTP mix',rxns*q['dntp_ul_per_rxn']*over,prices['dntp'],[3,4],'µL; manufacturer concentration.')
 add('Target primers',primer_pmol*over,prices['primer'],[3,5],'pmol;30bases and10nmol delivered per primer assumed; recurring use only.')
 add('Library end repair ligation and PCR',libs*over,prices['library_kit'],[6,7],'Complete kit preparation equivalents; no additional indexing PCR mix charged.')
 add('Adapters USER and unique dual indexes',libs,prices['indexes'],[8],'One complete well per library;96 distinct indexes PER RUN, fresh reagent for every library.')
 add('Cleanup beads',total_beads*over,prices['beads'],[7,9],'µL; target cleanup ratio assumed; kit cleanup volumes from manual.')
 add('Fluorometric quantification',qubits*over,prices['qubit'],[10],'Input DNA, each cleaned target pool, final library, run pools and standards; assumed schedule.')
 add('Qubit tubes',qubits,prices['qubit_tube'],[11],'One tube per measurement.')
 add('TapeStation QC',math.ceil((libs+runs)/12),68,[9],'Full12-position service blocks; all final libraries and run pools. Internal price anchor.')
 add('Run-pool qPCR',qpcr*over,prices['qpcr'],[12],'27 reactions/run:6 standards in triplicate, blank triplicate, pool at2 dilutions in triplicate.')
 add('Filter tips',tip_count,prices['tips'],[13],'Explicit task count; handling allowance, not empirically measured.')
 add('PCR plates and seals',plates,prices['plate']+prices['seal'],[13],'Whole plates rounded up separately for reaction/cleanup stages.')
 add('Storage and mixing tubes',2*libs+20*runs,prices['tube'],[13],'Assumed2/library plus20/run; Qubit tubes separate.')
 add('Minor buffers ethanol and water',libs,q['minor_buffer_allowance_per_library'],[],'Unquoted allowance; includes wash ethanol, TE/water/dilution buffer. Sensitivity0.5–2USD/library.')
 add('Methylation standard DNA',len(q['methylation_levels'])*q['control_input_ng']*runs*over,prices['controls'],[1,16,17],'ng;seven symmetric levels,100ng input/level assumed; blank contains no DNA.')
 add('PhiX',2*runs,prices['phix'],[19,20,21],'µL stock; fresh preparation/run. Historical price; no credit for reusing diluted stock.')
 add('Sequencing runs',runs,prices['run'],[14,18],'Full MiSeq v2 500-cycle run charge; conversion and run fees external academic anchors.')
 add('Demultiplexing service',runs,prices['multiplexing'],[14],'Posted multiplexing fee conservatively charged once/run; confirm invoice scope.')
 add('Hands-on labour',hours,q['labor_hourly'],[15],'Task durations assumed; rate published. Excludes labour inside purchased services.')
 total=sum(x['total'] for x in rows)
 return {'samples':n,'runs':runs,'controls':runs*controls,'libraries':libs,'libraries_per_run':perrun,'raw_pairs_per_library':reads,'reaction_count':rxns,'hours':hours,'labor_tasks':labor_tasks,'plates':plates,'components':rows,'total':total,'per_processed':total/n,'epic_ratio':q['epic']/(total/n),'per_reportable_90pct_sensitivity':total/(n*.9)}

def main():
    global p,prices
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config/assay_cost.json')
    parser.add_argument('--out-dir',required=True)
    args=parser.parse_args();config=json.loads(Path(args.config).read_text())
    p=config['parameters'];prices=config['unit_prices'].copy()
    out=Path(args.out_dir)
    if out.exists() and any(out.iterdir()):raise ValueError('Output directory must be empty')
    out.mkdir(parents=True,exist_ok=True)
    scenarios=[calculate(n) for n in config['batch_sizes']]
    sensitivity=[]
    for row in config['sensitivity']:
        changes=row['changes'].copy();old_run=prices['run']
        if 'run_fee' in changes:prices['run']=changes.pop('run_fee')
        value=calculate(config['batch_sizes'][-1],**changes)
        prices['run']=old_run
        sensitivity.append({'scenario':row['scenario'],'runs':value['runs'],'per_processed':value['per_processed']})
    result={'parameters':p,'unit_prices':prices,'references':config['references'],
            'scenarios':scenarios,'sensitivity':sensitivity,'interpretation':config['interpretation'],
            'config_sha256':sha256(args.config)}
    (out/'cost_model.json').write_text(json.dumps(result,indent=2)+'\n')
    with (out/'components.tsv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['samples']+list(scenarios[0]['components'][0]),delimiter='\t',lineterminator='\n');writer.writeheader()
        for batch in scenarios:
            for row in batch['components']:writer.writerow({'samples':batch['samples'],**row})

if __name__=='__main__':main()
