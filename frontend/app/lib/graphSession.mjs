// Only responses from the same-origin graph session identify a standalone viewer.
// An upstream outage is not sign-out; let startup show its existing retry screen.
export async function bootstrapGraphSession({signal, fetchImpl = fetch} = {}) {
  const read = path => fetchImpl(path, {signal, cache: 'no-store', credentials: 'same-origin'});
  const auth = await read('/api/matrix/session-auth');
  if (auth.status === 401) return {token: '', name: '', authorId: null};
  if (!auth.ok) throw new Error('Graph session unavailable');
  const issued = await auth.json();
  if (typeof issued.token !== 'string' || !issued.token) throw new Error('Invalid graph session');
  const accountResponse = await read('/api/account');
  if (!accountResponse.ok) throw new Error('Graph account unavailable');
  const {account} = await accountResponse.json();
  const identities = account?.identities;
  if (!Array.isArray(identities)) throw new Error('Invalid graph account');
  const orcid = identities.find(identity => identity.provider === 'orcid');
  const user = orcid || identities.find(identity => identity.provider === 'google');
  if (!user) throw new Error('Graph identity unavailable');
  let authorId = null;
  if (orcid) {
    const mappingResponse = await read('/api/node-mapping');
    if (!mappingResponse.ok) throw new Error('Graph profile unavailable');
    const mapping = await mappingResponse.json();
    if (mapping.authorId != null) {
      const id = Number(mapping.authorId);
      if (!Number.isSafeInteger(id) || id === 0) throw new Error('Invalid graph profile');
      authorId = String(id);
    }
  }
  return {token: issued.token, name: String(user.name || ''), authorId};
}
