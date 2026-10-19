import {useState} from 'react';
import {loadResearchContext, EMPTY_RESEARCH_CONTEXT} from '../lib/researchContext.mjs';
import {BRIDGE_MSG, inBridgeIframe, postToBridge, graphPagePath} from '../lib/bridgeHost';

export default function ResearchContextEditor({value,onChange}) {
 const current=value || EMPTY_RESEARCH_CONTEXT;
 const [loading,setLoading]=useState(false),[notice,setNotice]=useState('');
 const refresh=async()=>{
  setLoading(true);setNotice('');
  try {onChange(await loadResearchContext());}
  catch {setNotice('Interests could not be loaded. Please retry.');}
  finally {setLoading(false);}
 };
 return <section className="personal-context" aria-labelledby="personal-context-title">
  <h3 id="personal-context-title" className="context-section-label">Research interests</h3>
  {!!current.topics?.length && <><p className="you-note">{current.topics.join(' · ')}</p><label className="personal-context-toggle"><input type="checkbox" checked={current.useInterests !== false} onChange={e=>onChange({...current,useInterests:e.target.checked})}/>Use these interests</label></>}
  <div className="personal-context-actions">
   {inBridgeIframe() ? <button className="you-more" type="button" onClick={()=>postToBridge({type:BRIDGE_MSG.EDIT_INTERESTS})}>Edit interests</button> : <a className="you-more" href={graphPagePath('/?open=interests')} target="_blank" rel="noopener noreferrer">Edit interests ↗</a>}
   <button type="button" className="you-more" disabled={loading} onClick={refresh}>{loading?'Loading…':'Use latest interests'}</button>
  </div>
  {notice && <p role="alert" className="you-note">{notice}</p>}
 </section>;
}
