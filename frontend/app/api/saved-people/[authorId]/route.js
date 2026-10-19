import {graphAccountProxy} from '../../../lib/graphAccountProxy.mjs';
export async function DELETE(request,{params}) {
  const {authorId} = await params;
  if (!/^-?[1-9]\d*$/.test(authorId)) return Response.json({message:'Invalid researcher ID'},{status:400});
  return graphAccountProxy(request,`/api/saved-people/${encodeURIComponent(authorId)}`);
}
