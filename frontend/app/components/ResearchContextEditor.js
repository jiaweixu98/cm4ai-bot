import { useState } from 'react';
import { loadResearchContext, EMPTY_RESEARCH_CONTEXT } from '../lib/researchContext.mjs';

const RELATIONSHIPS = {met:'Have met',collaborated:'Have worked with',know:'Know personally',interested:'Would like to connect'};

export default function ResearchContextEditor({ value, onChange }) {
 const current=value || EMPTY_RESEARCH_CONTEXT;
 const [loading,setLoading]=useState(false),[notice,setNotice]=useState('');
 const update=patch=>onChange({...current,...patch});
 const refresh=async()=>{
  setLoading(true);setNotice('');
  try{onChange(await loadResearchContext());setNotice('Latest profile loaded into this conversation.');}
  catch{setNotice('Could not load your profile. Your current context has been kept.');}
  finally{setLoading(false);}
 };
 return <section className="personal-context" aria-labelledby="personal-context-title">
  <h3 id="personal-context-title" className="context-section-label">Your research context</h3>
  <p className="you-note">A snapshot of your saved profile. Changes here apply only to this conversation.</p>
  <label className="context-picker-label">Research interests<textarea rows={3} maxLength={2000} value={current.interests} placeholder="Describe your research interests for this conversation" onChange={e=>update({interests:e.target.value})}/></label>
  <label className="personal-context-toggle"><input type="checkbox" checked={current.useInterests} onChange={e=>update({useInterests:e.target.checked})}/>Use these interests in this conversation</label>
  <div className="personal-context-heading"><h4>People & connections</h4><span>{current.contacts.filter(p=>p.included).length} included</span></div>
  {!current.contacts.length&&<p className="personal-context-empty">Add people you know or have met in My graph profile. No graph node is required.</p>}
  <ul className="personal-context-list">{current.contacts.map(p=><li key={p.id}>
   <label className="personal-context-toggle"><input type="checkbox" checked={p.included} onChange={e=>update({contacts:current.contacts.map(other=>other.id===p.id?{...other,included:e.target.checked}:other)})}/><strong>{p.name}</strong></label>
   <p className="you-note">{RELATIONSHIPS[p.relationship]}{p.note?` · ${p.note}`:''}</p>
  </li>)}</ul>
  <p className="you-note">Only checked information is sent with your next message. These are your own descriptions, not verified relationships or recommendations.</p>
  <div className="personal-context-actions"><a className="you-more" href="/?open=profile" target="_blank" rel="noopener noreferrer">Edit My graph profile ↗</a><button type="button" className="you-more" disabled={loading} onClick={refresh}>{loading?'Loading…':'Use latest profile'}</button></div>
  <p className="you-note">Editing your profile does not change this conversation until you choose “Use latest profile”.</p>
  {notice&&<p role="status" className="you-note">{notice}</p>}
 </section>;
}
