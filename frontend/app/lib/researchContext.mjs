export const EMPTY_RESEARCH_CONTEXT = {topics:[],interests:'',useInterests:true,contacts:[]};
export function topicContext(topics, useInterests = true) {
 const clean=[...new Set((Array.isArray(topics) ? topics : []).filter(t => typeof t === 'string' && t.trim() && t.length <= 200))].slice(0,8);
 return {topics:clean,interests:clean.join('; '),useInterests,contacts:[]};
}
export async function loadResearchContext() {
 const response=await fetch('/api/research-interests',{cache:'no-store'});
 if(response.status===401)return structuredClone(EMPTY_RESEARCH_CONTEXT);
 if(!response.ok)throw new Error('Research interests unavailable');
 const data=await response.json();
 return topicContext(data.topics);
}
