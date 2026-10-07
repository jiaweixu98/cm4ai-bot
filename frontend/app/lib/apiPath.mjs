// MATRIX is mounted at /matrix in production and at / in local development.
// Keep browser requests on the app's mount, away from Graph's /api routes.
export function matrixApiPath(path, pathname = typeof window === 'undefined' ? '/' : window.location.pathname) {
  return pathname === '/matrix' || pathname.startsWith('/matrix/') ? `/matrix${path}` : path;
}
