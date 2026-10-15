export const EMPTY_RESEARCH_CONTEXT = {interests:'',useInterests:true,contacts:[]};
export async function loadResearchContext() {
 const response=await fetch('/api/research-context',{cache:'no-store'});
 if(response.status===401)return structuredClone(EMPTY_RESEARCH_CONTEXT);
 if(!response.ok)throw new Error('Research context unavailable');
 const data=await response.json();
 return data.profile;
}
