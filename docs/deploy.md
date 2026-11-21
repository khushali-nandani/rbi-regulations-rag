# Deploying to Hugging Face Spaces

The app ships as a Docker image (corpus, index and embedding models are baked in at build time), so a
Docker Space can run it as is. The free CPU tier (2 vCPU, 16 GB RAM) is plenty: the container uses
about 270 MB.

## One-time setup (about 10 minutes)

1. **Create a Space** at <https://huggingface.co/new-space>: SDK **Docker** (blank template), visibility
   **Public**, hardware **CPU basic (free)**.
2. **Create an access token** at <https://huggingface.co/settings/tokens> with **write** access.
3. **Add Space secrets** (Space → Settings → Variables and secrets → *New secret*):

   | Name | Value |
   |---|---|
   | `ANTHROPIC_API_KEY` | your Anthropic key |
   | `API_KEYS` | one or more keys that visitors will type into the demo page (comma separated) |
   | `ADMIN_API_KEYS` | a different key, only for `/traces*` |

   And optional **variables** (not secrets): `MAX_SPEND_USD=1` and `RATE_LIMIT_PER_MIN=10`.
4. **Cap spend at the source.** In the Anthropic console, set a monthly spend limit on the key. The
   in-app limit resets whenever the Space restarts, so the console limit is the real safety net.

## Deploy

```bash
git status                       # must be clean: the Space gets exactly what HEAD contains
HF_TOKEN=hf_xxx scripts/deploy_space.sh <hf_username>/<space_name>
```

The first build takes a few minutes (it ingests the PDFs, builds the index and downloads the embedding
models inside Hugging Face's builder). Watch it on the Space page. When it says **Running**, open
`https://<hf_username>-<space_name>.hf.space`.

## Check it

```bash
curl -s https://<hf_username>-<space_name>.hf.space/health
curl -s -X POST https://<hf_username>-<space_name>.hf.space/ask \
  -H "X-API-Key: <one of API_KEYS>" -H "content-type: application/json" \
  -d '{"question": "What is the shortest cooling-off period for a digital loan?"}'
```

## Sharing the demo safely

- The demo page is public but every question needs an API key, so nobody can spend your budget
  without one. Give recruiters a **dedicated demo key** (one entry in `API_KEYS`) and rotate it by
  changing the secret if it leaks.
- Rate limiting is per key (10 requests/minute by default), so one visitor can exhaust the minute for
  everyone sharing a key. That is acceptable for a demo; it is not multi-tenant auth.
- Free Spaces sleep after inactivity and restart on the next visit (a short cold start). Traces, the LLM
  cache and the in-app spend counter are ephemeral.

## Updating

Commit to `main`, then run the deploy script again. It force-pushes to the Space repo, which is
generated output and holds nothing you need to keep.
