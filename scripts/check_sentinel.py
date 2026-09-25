"""Diagnose SentinelPulse connectivity and historical coverage."""
import json
import subprocess
import sys
import urllib.request

def get_sp_key():
    with open('/Users/manishkumar/Desktop/ml-service2.0/.env') as f:
        for line in f:
            if line.startswith('SENTINEL_PULSE_API_KEY='):
                return line.strip().split('=', 1)[1]
    return ''

def get_sp_url():
    with open('/Users/manishkumar/Desktop/ml-service2.0/.env') as f:
        for line in f:
            if line.startswith('SENTINEL_PULSE_URL='):
                return line.strip().split('=', 1)[1]
    return 'http://localhost:3001'

def fetch(url, key):
    req = urllib.request.Request(url, headers={'Authorization': f'Bearer {key}'})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except Exception as e:
        return {'error': str(e)}

key = get_sp_key()
base = get_sp_url()
print(f'Base URL: {base}')
print(f'API key: length={len(key)}, set={len(key)>0}')
print()

# Health
h = fetch(f'{base}/health', key)
print(f'Health: {h.get("status")}')

# Market context
mc = fetch(f'{base}/api/v1/alphaforge/context/market', key)
print(f'Market context success: {mc.get("success")}')
data = mc.get('data', {})
print(f'  regime: {data.get("regime")}')
top_events = data.get('top_events', [])
print(f'  top_events: {len(top_events)}')
sentiment = data.get('overall_sentiment', {})
print(f'  overall sentiment: {sentiment.get("overall")}')

# Training samples
ts = fetch(f'{base}/api/v1/ml/training/samples?limit=5', key)
samples = ts.get('data', []) if isinstance(ts.get('data'), list) else ts.get('data', {}).get('samples', [])
total = ts.get('meta', {}).get('total_count', 0)
print(f'\nTraining samples: total_count={total}, returned={len(samples)}')

# Historical reactions
hr = fetch(f'{base}/api/v1/ml/historical-reactions?limit=5', key)
reactions = hr.get('data', [])
hr_total = hr.get('meta', {}).get('total_count', 0)
print(f'Historical reactions: total_count={hr_total}, returned={len(reactions)}')

# High impact events
hi = fetch(f'{base}/api/v1/alphaforge/high-impact-events?limit=5', key)
events = hi.get('data', [])
hi_total = hi.get('meta', {}).get('total_count', len(events))
print(f'High impact events: total={hi_total}, returned={len(events)}')

# News context for RELIANCE
nc = fetch(f'{base}/api/v1/alphaforge/news-context/RELIANCE', key)
print(f'\nNews context RELIANCE: success={nc.get("success")}')
if nc.get('success'):
    d = nc.get('data', {})
    print(f'  news_impact_score: {d.get("news_impact_score")}')
    print(f'  computed_at: {d.get("computed_at")}')
    print(f'  velocity_1h: {d.get("velocity_metrics", {}).get("articles_1h")}')
    print(f'  velocity_24h: {d.get("velocity_metrics", {}).get("articles_24h")}')
else:
    print(f'  error: {nc.get("error")}')

# News feature vector for RELIANCE
fv = fetch(f'{base}/api/v1/ml/features/asset/RELIANCE?limit=5', key)
features = fv.get('data', [])
fv_total = fv.get('meta', {}).get('total_count', len(features))
print(f'\nNews features RELIANCE: total={fv_total}, returned={len(features)}')
if features:
    f0 = features[0]
    print(f'  sample: featureType={f0.get("featureType")}, computedAt={f0.get("computedAt")}')

# CONCLUSION
print('\n=== SUMMARY ===')
print(f'SentinelPulse REACHABLE: {h.get("status") == "alive"}')
print(f'AUTH WORKING: {mc.get("success", False)}')
print(f'TRAINING SAMPLES: {total}')
print(f'HISTORICAL REACTIONS: {hr_total}')
print(f'FEATURE VECTORS: {fv_total}')
print(f'NEWS COVERAGE VERDICT: {"EVIDENCE_PRESENT" if (total > 0 or hr_total > 0) else "NO_HISTORICAL_SAMPLES"}')
