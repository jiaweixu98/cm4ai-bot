/** Standalone Ask KG uses the same signed-in account API as the graph. */
export async function accountSavedPeople(path = '', options = {}) {
  const response = await fetch(`/api/saved-people${path}`, {
    cache: 'no-store', ...options,
    headers: { Accept: 'application/json', 'Content-Type': 'application/json', ...options.headers },
  });
  if (!response.ok) throw new Error(response.status === 401
    ? 'Sign in on the graph to save people to your account.'
    : 'Could not update saved people. Please try again.');
  return response.json();
}
