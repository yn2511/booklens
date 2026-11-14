// BookLens DLS proxy: booklens.cloud -> read365.edunet.net (서울문정초등학교 장서 검색)
const SEARCH_URL = 'https://read365.edunet.net/alpasq/api/search';
const STATE_URL = 'https://read365.edunet.net/alpasq/api/search/book/state';
const PROV = 'B10';
const NEIS = 'B100000919';
const SCHOOL = '서울문정초등학교';
const ALLOWED_ORIGINS = ['https://booklens.cloud', 'https://www.booklens.cloud'];
const CACHE_TTL_SECONDS = 600; // 10분
const buckets = new Map();     // per-isolate best-effort rate limit

function corsHeaders(origin) {
  const o = ALLOWED_ORIGINS.includes(origin) ? origin : ALLOWED_ORIGINS[0];
  return {
    'Access-Control-Allow-Origin': o,
    'Access-Control-Allow-Methods': 'GET, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Vary': 'Origin'
  };
}
function json(obj, status, origin, extra) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: Object.assign({ 'Content-Type': 'application/json; charset=utf-8' }, corsHeaders(origin), extra || {})
  });
}
function normTitle(t) {
  return (t || '').replace(/\s+/g, ' ').trim().slice(0, 80);
}
function rateOk(ip) {
  const now = Date.now();
  let b = buckets.get(ip);
  if (!b || now > b.reset) { b = { count: 0, reset: now + 60000 }; buckets.set(ip, b); }
  b.count += 1;
  return b.count <= 30; // 30 req/min/IP
}
async function fetchWithTimeout(url, opts, ms) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), ms);
  try {
    return await fetch(url, Object.assign({}, opts, { signal: ctrl.signal }));
  } finally {
    clearTimeout(timer);
  }
}
async function loanStatusOf(bookKey) {
  try {
    const r = await fetchWithTimeout(`${STATE_URL}?bookKey=${bookKey}&neisCode=${NEIS}&provCode=${PROV}`, null, 5000);
    const j = await r.json();
    const d = j && j.data;
    if (!d) return null;
    if (typeof d === 'string') return d;
    return d.status || d.loanStatusDesc || d.statusDesc || d.loanYn || d.stateDesc || null;
  } catch (_) { return null; }
}
export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const origin = request.headers.get('Origin') || '';
    if (request.method === 'OPTIONS') return new Response(null, { status: 204, headers: corsHeaders(origin) });
    if (url.pathname !== '/dls-search') {
      return json({ ok: true, service: 'booklens-dls-proxy', usage: '/dls-search?title=책제목' }, 200, origin);
    }
    const title = normTitle(url.searchParams.get('title'));
    if (!title) return json({ found: false, total: 0, books: [], error: 'empty_title' }, 400, origin);
    const ip = request.headers.get('CF-Connecting-IP') || 'anon';
    if (!rateOk(ip)) return json({ found: false, total: 0, books: [], error: 'rate_limited' }, 429, origin);

    const cacheKey = new Request(url.origin + '/dls-search?title=' + encodeURIComponent(title.toLowerCase()));
    const cached = await caches.default.match(cacheKey);
    if (cached) {
      return new Response(cached.body, { status: 200, headers: Object.assign({ 'Content-Type': 'application/json; charset=utf-8', 'X-Cache': 'HIT' }, corsHeaders(origin)) });
    }

    try {
      const body = {
        searchKeyword: title, page: 1, display: 5,
        neisCode: [NEIS], provCode: PROV, schoolName: SCHOOL,
        coverYn: 'N', facet: 'N'
      };
      const r = await fetchWithTimeout(SEARCH_URL, {
        method: 'POST', headers: { 'Content-Type': 'application/json', 'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36', 'Accept': 'application/json, text/plain, */*', 'Referer': 'https://read365.edunet.net/PureScreen/SchoolSearch?schoolName=%EC%84%9C%EC%9A%B8%EB%AC%B8%EC%A0%95%EC%B4%88%EB%93%B1%ED%95%99%EA%B5%90&provCode=B10&neisCode=B100000919', 'Origin': 'https://read365.edunet.net' }, body: JSON.stringify(body)
      }, 8000);
      if (!r.ok) throw new Error('dls_status_' + r.status);
      const j = await r.json();
      const data = (j && j.data) || {};
      const list = data.bookList || [];
      const books = list.slice(0, 5).map(b => ({
        title: b.title, author: b.author, publisher: b.publisher, pubYear: b.pubYear,
        callNo: b.callNo, location: b.locationName, isbn: b.isbn
      }));
      if (list[0] && list[0].bookKey) {
        const st = await loanStatusOf(list[0].bookKey);
        if (st) books[0].loanStatus = st;
      }
      const out = { found: books.length > 0, total: data.totalCount || 0, books };
      const res = json(out, 200, origin, { 'Cache-Control': 'public, max-age=' + CACHE_TTL_SECONDS, 'X-Cache': 'MISS' });
      ctx.waitUntil(caches.default.put(cacheKey, res.clone()));
      return res;
    } catch (e) {
      return json({ found: false, total: 0, books: [], error: 'dls_unavailable', detail: String(e && e.message || e).slice(0, 200) }, 502, origin);
    }
  }
};
