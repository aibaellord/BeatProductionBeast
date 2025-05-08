# BeatProductionBeast API Reference

## Endpoints

### POST /generate-beat/
- Generate a new beat with a given style and consciousness level.
- Parameters: style (str), consciousness_level (int)

### POST /remix/
- Remix an uploaded audio file with an optional preset.
- Parameters: file (audio), preset (str, optional)

### GET /revenue/dashboard/
- Get a summary of all revenue streams.

### POST /investment/option/
- Submit an investment for premium features or support.
- Parameters: amount (float), user_id (str)

### GET /analytics/
- Get analytics on users, beats generated, revenue, active sessions, top styles, and conversion rate.

### POST /marketing-campaign/
- Launch an automated marketing campaign for a beat across selected platforms.
- Parameters: beat_id (str), platforms (list), campaign_type (str, default: "launch")

### GET /marketing-analytics/
- Retrieve aggregated marketing analytics across all channels.

### POST /remix-challenge/
- Launch a fully automated remix challenge.
- Parameters: challenge_name (str), source_track (str), deadline (str)

### POST /batch-release/
- Schedule and distribute a batch of tracks automatically.
- Parameters: batch (list of track IDs or names)

### POST /auto-curation/
- AI/ML-powered auto-curation of user portfolio for best exposure.
- Parameters: user_id (str)

### POST /influencer-collab/
- Initiate automated influencer collaboration and cross-promotion.
- Parameters: beat_id (str), influencer_handle (str)

### POST /auto-translation/
- AI-powered auto-translation for global reach (UI, metadata, lyrics).
- Parameters: text (str), target_lang (str)

### POST /sync-marketplace/upload/
- Upload a beat for sync licensing. AI auto-tags and generates instant license.
- Parameters: beat_file (file), metadata (dict)

### POST /sync-marketplace/match/
- Match a project (video, ad, etc.) to the best beats in the marketplace.
- Parameters: project_file (file), project_type (str)

### POST /sync-marketplace/license/
- Instantly license a beat, auto-generate contract and handle payment.
- Parameters: beat_id (str), buyer_info (dict)

### GET /sync-marketplace/listings/
- List all available beats for sync licensing.

### GET /sync-marketplace/analytics/
- Get analytics for buyers and sellers in the sync marketplace.

---

See OpenAPI docs at `/docs` for full details.
