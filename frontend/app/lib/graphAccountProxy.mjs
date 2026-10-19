// The local chat and Graph run on separate ports. Validate the browser's origin
// here before forwarding account mutations; do not relax Graph's CSRF check.
export async function graphAccountProxy(request, path, {fetchImpl=fetch,
  graphUrl=process.env.GRAPH_BACKEND_URL || (process.env.NODE_ENV === 'production'
    ? 'http://127.0.0.1:5173' : 'http://127.0.0.1:4173')} = {}) {
  const mutation = !['GET','HEAD'].includes(request.method);
  const origin = request.headers.get('origin');
  if (mutation) {
    try {
      const source = new URL(origin);
      if (!['http:','https:'].includes(source.protocol) || source.origin !== origin
        || source.host !== (request.headers.get('host') || new URL(request.url).host)) throw new Error('Origin');
    } catch {
      return Response.json({message:'Invalid request origin'}, {status:403});
    }
  }
  const target = new URL(path, graphUrl);
  const headers = new Headers({Accept:'application/json'});
  const cookie = request.headers.get('cookie');
  if (cookie) headers.set('cookie',cookie);
  if (mutation) {
    headers.set('content-type','application/json');
    const local = target.hostname === '127.0.0.1' && target.port === '4173';
    headers.set('origin',local ? target.origin : origin);
  }
  try {
    const response = await fetchImpl(target, {method:request.method,headers,
      ...(mutation ? {body:await request.text()} : {}),cache:'no-store',
      signal:AbortSignal.any([request.signal,AbortSignal.timeout(8000)])});
    return new Response(response.body,{status:response.status,headers:{
      'Content-Type':response.headers.get('content-type') || 'application/json', 'Cache-Control':'no-store'}});
  } catch {
    return Response.json({message:'Saved people are temporarily unavailable'}, {status:502});
  }
}
