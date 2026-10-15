"""Generate a separate driver profile and audit the installed Maestro bindings."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maestro',type=Path,default=Path(r'C:\Program Files (x86)\Steam\steamapps\common\Maestro'))
    args=parser.parse_args();root=Path(__file__).parent
    src=args.maestro/'WindowsNoEditor/Maestro/Config/SteamVRBindings'
    manifest=json.loads((src/'steamvr_manifest.json').read_text())
    binding=json.loads((src/'oculus_touch.json').read_text())
    binding.update(controller_type='fh_skeletal',app_key='steam.app.2701700',name='Frame Hands optical skeletons for Maestro',description='Native hand skeletons; optional controller fallback.')
    sets=binding['bindings']['/actions/main']
    for entry in sets['sources']:
        entry['path']=entry['path'].replace('/input/x','/input/a').replace('/input/y','/input/b')
    sets['sources'].append(dict(mode='button',path='/user/hand/right/input/system',inputs=dict(click=dict(output='/actions/main/in/JJHMD_Right_System'))))
    profile=dict(jsonid='input_profile',controller_type='fh_skeletal',device_class='TrackedDeviceClass_Controller',resource_root='fh_skeletal',driver_name='fh_skeletal',input_bindingui_mode='controller_handed',should_show_binding_errors=True,input_source={},default_bindings=[dict(app_key='steam.app.2701700',binding_url='maestro.json'),dict(app_key='openvr.component.vrcompositor',binding_url='dashboard.json')],remapping='{fh_skeletal}/input/remapping.json')
    sources=profile['input_source']
    for name in ['trigger','grip']:sources[f'/input/{name}']=dict(type='trigger',click=True,value=True)
    for name in ['a','b','application_menu','system']:sources[f'/input/{name}']=dict(type='button',click=True)
    sources['/input/joystick']=dict(type='joystick',click=True)
    for pose in ['raw','tip','grip']:sources[f'/pose/{pose}']=dict(type='pose')
    for side in ['left','right']:sources[f'/input/skeleton/{side}']=dict(type='skeleton',side=side,skeleton=f'/skeleton/hand/{side}')
    sources['/output/haptic']=dict(type='vibration')
    remapping=dict(to_controller_type='fh_skeletal',layouts=[dict(priority=100,from_controller_type='oculus_touch',simulate_controller_type=True,simulate_render_model=False,simulate_HMD=False,autoremappings=[dict(**{'from':'/user/hand/left/input/x','to':'/user/hand/left/input/a','mirror':False}),dict(**{'from':'/user/hand/left/input/y','to':'/user/hand/left/input/b','mirror':False})])])
    dashboard=dict(controller_type='fh_skeletal',app_key='openvr.component.vrcompositor',name='Frame Hands pinch dashboard',description='Controller fallback mode only',bindings={'/actions/lasermouse':dict(sources=[],poses=[])})
    for side in ['left','right']:
        dashboard['bindings']['/actions/lasermouse']['poses'].append(dict(path=f'/user/hand/{side}/pose/raw',output='/actions/lasermouse/in/Pointer'))
        dashboard['bindings']['/actions/lasermouse']['sources'].append(dict(path=f'/user/hand/{side}/input/trigger',mode='trigger',inputs=dict(click=dict(output='/actions/lasermouse/in/leftclick'))))
    dest=root/'driver/fh_skeletal';(dest/'resources/input').mkdir(parents=True,exist_ok=True)
    (dest/'driver.vrdrivermanifest').write_text(json.dumps(dict(name='fh_skeletal',alwaysActivate=True,resourceOnly=False,directory=''),indent=2)+'\n')
    for name,obj in [('fh_skeletal_profile',profile),('maestro',binding),('remapping',remapping),('dashboard',dashboard)]:
        (dest/f'resources/input/{name}.json').write_text(json.dumps(obj,indent=2)+'\n')
    outputs=set()
    def visit(x):
        if isinstance(x,dict):
            if 'output' in x:outputs.add(x['output'].lower())
            for v in x.values():visit(v)
        elif isinstance(x,list):
            for v in x:visit(v)
    visit(binding)
    rows=[]
    for a in manifest['actions']:
        name=a['name'];bound=name.lower() in outputs
        if '/tracker_' in name:reason='Optional body/accessory tracker; not a hand input'
        elif 'open_console' in name:reason='Developer console; not gameplay'
        elif name.endswith(' axis'):reason='Unbound scalar alias in shipped controller bindings; vector2 stick action is bound'
        elif name.lower().endswith('right_system'):reason='Reserved system input; no synthesized gesture; use physical controller'
        elif a['type']=='vibration':reason='Bound output; bare hands have no haptic actuator'
        elif a['type']=='skeleton':reason='31 bones, both motion ranges, full optical finger input'
        elif a['type']=='pose':reason='Wrist pose in PC raw tracking space at capture time'
        elif a['type']=='vector2' or '_Stick' in name:reason='Controller fallback: virtual stick in --controls mode'
        else:reason='Controller fallback in --controls mode; native gameplay uses skeleton'
        rows.append(dict(name=name,type=a['type'],binding_present=bound,coverage=reason))
    evidence=dict(verified='2026-10-03',game_appid=2701700,game_build=24847949,manifest_sha256=hashlib.sha256((src/'steamvr_manifest.json').read_bytes()).hexdigest(),actions=rows,limits='Static binding audit does not prove which backend or actions gameplay uses; native OpenXR and worn Maestro trials required.')
    (root/'maestro-input-audit.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(dict(actions=len(rows),bound=sum(r['binding_present'] for r in rows))))


if __name__=='__main__':main()
