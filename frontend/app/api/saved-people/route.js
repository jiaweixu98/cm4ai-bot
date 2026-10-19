import {graphAccountProxy} from '../../lib/graphAccountProxy.mjs';
export const GET = request => graphAccountProxy(request,'/api/saved-people');
export const PUT = request => graphAccountProxy(request,'/api/saved-people');
