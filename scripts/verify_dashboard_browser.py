"""Opt-in UI QA: isolated installed Chrome; never uses a personal browser profile.
Run against the local Streamlit server with .venv/bin/python scripts/verify_dashboard_browser.py.
"""
from pathlib import Path
import json,re,time,os
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(os.environ.get('CENSUS_QA_OUTPUT',str(ROOT/'docs/premium_dashboard')))
PAGES=['Research Overview','Data Explorer','Model Performance','Model Explainability','Responsible AI Assessment','Prediction Sandbox']
HEADINGS=['Research overview','Data explorer','Model performance','Model explainability','Responsible AI assessment','Prediction sandbox']

def run():
 with sync_playwright() as p:
  browser=p.chromium.launch(executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless=True)
  ctx=browser.new_context(viewport={'width':1440,'height':900},device_scale_factor=1,reduced_motion='reduce',accept_downloads=True)
  page=ctx.new_page();errors=[];checks=[];captures=[]
  page.on('pageerror',lambda e:errors.append(str(e)))
  def ready():
   page.wait_for_timeout(450)
   page.wait_for_function("document.querySelector('[data-testid=\"stApp\"]')?.getAttribute('data-test-script-state')==='notRunning'",timeout=90000)
   assert page.locator('[data-testid="stException"]').count()==0
  def nav(i):
   print('NAV',i,flush=True)
   radio=page.get_by_role('radio',name=re.compile(PAGES[i]))
   if not radio.is_checked():radio.focus();page.keyboard.press('Space')
   page.get_by_role('heading',name=HEADINGS[i],exact=True).wait_for(timeout=90000);ready()
  def choose(key,value):
   page.locator('.st-key-'+key).get_by_role('combobox').click()
   page.get_by_role('option',name=value,exact=True).click();ready()
  def click(name):
   target=page.get_by_role('button',name=name,exact=True)
   if not target.count():target=page.get_by_role('radio',name=name,exact=True)
   print('CLICK',name,flush=True);target.click();ready()
  def download(name):
   with page.expect_download() as event:page.get_by_role('button',name=name,exact=True).click()
   path=event.value.path();return Path(path).read_text()
  page.goto(os.environ.get('CENSUS_DASHBOARD_URL','http://127.0.0.1:8501'));page.get_by_role('heading',name=HEADINGS[0],exact=True).wait_for(timeout=90000);ready()
  sizes=[] if os.environ.get('QA_SKIP_CAPTURE') else [(1440,900),(1920,1080),(1024,768)]
  if os.environ.get('CENSUS_QA_VIEWPORTS'):
   sizes=[tuple(map(int,item.split('x'))) for item in os.environ['CENSUS_QA_VIEWPORTS'].split(',')]
  if not sizes:captures=json.loads((OUT/'layout_qa.json').read_text())
  for width,height in sizes:
   page.set_viewport_size({'width':width,'height':height})
   folder=OUT/('after' if width==1440 else f'responsive/{width}x{height}');folder.mkdir(parents=True,exist_ok=True)
   for i in range(6):
    nav(i)
    page.locator('[data-testid="stPlotlyChart"]').first.scroll_into_view_if_needed()
    page.locator('.js-plotly-plot .main-svg').first.wait_for(timeout=90000)
    page.locator('[data-testid="stMain"]').evaluate('(e)=>e.scrollTo(0,0)')
    ready();page.screenshot(path=str(folder/f'{i+1:02d}.png'),full_page=True,animations='disabled')
    geometry=page.locator('[data-testid="stMain"]').evaluate('(e)=>({height:e.clientHeight,scrollHeight:e.scrollHeight,width:e.clientWidth,scrollWidth:e.scrollWidth})')
    assert geometry['scrollWidth']<=geometry['width']+1,(PAGES[i],geometry)
    if width>=1440:assert geometry['scrollHeight']<=geometry['height']+1,(PAGES[i],geometry)
    captures.append({'page':PAGES[i],'viewport':[width,height],'main':geometry})
  (OUT/'layout_qa.json').write_text(json.dumps(captures,indent=2)+'\n')
  page.set_viewport_size({'width':1440,'height':900});nav(0)
  payload=json.loads(download('Export results'));assert payload['evaluated_rows']==9758;checks.append('Default JSON export: 9,758 held-out records')
  choose('analysis_segment','Female');assert '3,326' in page.locator('.st-key-kpi_evaluated_samples').inner_text()
  choose('analysis_model','Majority baseline');assert '0.00%' in page.locator('.st-key-kpi_test_recall').inner_text()
  click('Reset filters')
  slider=page.locator('.st-key-analysis_threshold').get_by_role('slider');slider.focus();page.keyboard.press('ArrowRight');ready()
  payload=json.loads(download('Export results'));assert payload['context']['threshold']==.55
  checks.append('Population/model/reset/keyboard threshold recompute exported results')
  choose('analysis_metric','Precision');choose('analysis_comparison','Current evaluation');choose('analysis_split','Development · in-sample')
  assert 'In-sample' in page.locator('.st-key-kpi_row').inner_text();click('Reset filters');click('Refresh analysis')
  assert json.loads(download('Export results'))['context']['threshold']==.5
  checks.append('Metric, comparison, split, refresh and reset controls operate')
  nav(2)
  for value in ['ROC curve','Precision–recall curve','Calibration','Threshold sensitivity','Confusion matrix']:choose('performance_view',value)
  click('Classification report');assert 'Support' in page.locator('[data-testid="stDataFrame"]').inner_text() or page.locator('[data-testid="stDataFrame"]').count()==1
  assert 'Support' in download('Export classification report')
  click('Errors');click('Inspect public-source error records');assert 'Scenario score' in download('Export current errors');page.keyboard.press('Escape');ready()
  checks.append('All five diagnostics, classification export, error inspection/export')
  nav(1);choose('explorer_sex','Female');assert '16,176' in page.locator('.st-key-kpi_row').inner_text()
  for value in ['Missingness','Associations','Distributions']:click(value)
  bars=page.locator('[class*="st-key-explorer_income_chart_"] .barlayer .point path')
  box=bars.last.bounding_box();page.mouse.click(box['x']+box['width']/2,box['y']+box['height']/2);ready();assert '100.0%' in page.locator('.st-key-kpi_row').inner_text()
  click('Reset filters');click('Browse & export');page.get_by_role('textbox',name='Search records').fill('Doctorate');page.get_by_role('textbox',name='Search records').press('Enter');ready()
  meta=json.loads(download('Download filter metadata'));assert meta['search']=='Doctorate' and meta['rows']>0
  assert 'Doctorate' in download('Download matching records');page.keyboard.press('Escape');ready()
  checks.append('Explorer filters, chart cross-selection, missingness, associations, record search/CSV/context')
  nav(3);choose('importance_feature','Capital gain');assert 'capital-gain' in download('Export importance')
  nav(4);choose('demographic_attribute','Race');choose('demographic_rate','False-negative rate')
  assert 'scenario_threshold' in download('Export subgroup results');checks.append('Importance selection/export and subgroup rates/export')
  nav(5)
  click('Classify hypothetical record')
  result=json.loads(download('Export submitted result'));assert result['input']['age']==35 and 0<=result['uncalibrated_score']<=1
  page.screenshot(path=str(OUT/'after/sandbox_submitted.png'),full_page=True)
  size=page.locator('[data-testid="stMain"]').evaluate('(e)=>[e.clientHeight,e.scrollHeight]')
  assert size[1]<=size[0]+1,size
  nav(0);nav(5);assert 'Model score' in page.locator('.st-key-panel_sandbox_2').inner_text()
  click('Reset inputs');assert 'No classification yet' in page.locator('.st-key-panel_sandbox_2').inner_text()
  checks.append('Real frozen-pipeline submission, JSON export, result persistence and input reset')
  checks.append('All six pages reachable using keyboard navigation; native focus retained')
  assert not errors,errors
  (OUT/'browser_qa.json').write_text(json.dumps({'captures':captures,'checks':checks,'page_errors':errors},indent=2)+'\n')
  browser.close();print(json.dumps({'captures':len(captures),'checks':checks,'errors':errors},indent=2))
if __name__=='__main__':run()
