# Railway + Discord setup

This project already includes a ready `Dockerfile`, so Railway can run it as a long-lived worker/container without any extra web server.

## What to prepare

You need:

- A Railway account
- A Discord application with a bot user
- At least one image-capable AI provider key

Recommended providers for `/draw`:

- `GEMINI_KEY` for the cheapest start
- `OPENAI_KEY` for more stable image generation

## Railway deployment

### 1. Log in to Railway CLI

```bash
railway login
```

If you prefer the Railway UI, you can also create the project in the dashboard and connect the GitHub repository there.

### 2. Create a new Railway project

From the repository root:

```bash
railway init
```

Or create a new empty project in the Railway dashboard and link it locally:

```bash
railway link
```

If you want the shortest path:

1. Log in to Railway
2. Create one empty project
3. Deploy this repository as a single service using the root `Dockerfile`
4. Paste the variables from the next section
5. Redeploy once

### 3. Add environment variables

Set these variables in Railway:

```env
DISCORD_BOT_TOKEN=your_discord_bot_token
LOGGING=True
REPLYING_ALL=False
DEFAULT_PROVIDER=gemini
DEFAULT_MODEL=auto
OPENAI_KEY=
CLAUDE_KEY=
GEMINI_KEY=your_gemini_key
GROK_KEY=
ADMIN_USER_IDS=
DISCORD_CHANNEL_ID=
REPLYING_ALL_DISCORD_CHANNEL_ID=
MAX_CONVERSATION_LENGTH=20
CONVERSATION_TRIM_SIZE=8
```

Notes:

- `DISCORD_BOT_TOKEN` is required
- For reliable `/draw`, configure `GEMINI_KEY` or `OPENAI_KEY`
- `DEFAULT_PROVIDER=gemini` is a good production default if you mainly want image generation
- `DISCORD_CHANNEL_ID` is optional and only used for the startup prompt behavior
- `REPLYING_ALL_DISCORD_CHANNEL_ID` is only needed if you enable `/replyall`
- `ADMIN_USER_IDS` is optional and only needed for restricted personas

### 4. Deploy

If the project is linked locally:

```bash
railway up
```

If you deploy from GitHub in the Railway UI, Railway should detect the `Dockerfile` automatically.

### 5. Check logs

After deployment, open Railway logs and confirm you see messages similar to:

- `Starting Discord AI Bot...`
- `Available providers: ...`
- `is now running!`

If the bot exits immediately, the most common cause is a missing `DISCORD_BOT_TOKEN` or provider key misconfiguration.

## What is already prepared in this repository

The repository is now prepared for Railway with:

- root `Dockerfile`
- root `railway.json`
- documented variables for Railway

That means Railway should build this as a worker-style service without adding any web server code.

## Discord bot setup

### 1. Create the application

Open:

- <https://discord.com/developers/applications>

Then:

1. Create a new application
2. Open the `Bot` section
3. Create a bot user
4. Copy the token into Railway as `DISCORD_BOT_TOKEN`

### 2. Enable intents

In the bot settings:

- Enable `MESSAGE CONTENT INTENT`

Why:

- Slash commands work without it
- The optional `/replyall` mode needs it

### 3. Invite the bot to your server

In `OAuth2 -> URL Generator` select:

- Scope: `bot`
- Scope: `applications.commands`

Recommended bot permissions:

- `View Channels`
- `Send Messages`
- `Embed Links`
- `Attach Files`
- `Read Message History`
- `Use Slash Commands`

Open the generated URL and add the bot to your server.

## First launch checklist

After the bot is invited and Railway deployment is healthy:

1. Wait for slash command sync after startup
2. Run `/help`
3. Run `/provider` and switch to `gemini` or `openai`
4. Run `/draw a cozy cyberpunk cat cafe at night`

If `/draw` fails:

- Verify that the current provider supports image generation
- Check the provider API key in Railway variables
- Review Railway deploy logs for provider-specific errors

## Recommended production setup

If the main use case is image generation, start with:

```env
DEFAULT_PROVIDER=gemini
DEFAULT_MODEL=auto
REPLYING_ALL=False
LOGGING=True
```

Then switch providers inside Discord with `/provider` when needed.

## Useful local commands

```bash
railway login
railway link
railway variables
railway up
railway logs
```
