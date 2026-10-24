// Example browser client for the public GPT-2 Space: waits for it to wake, asks the Game of Thrones model, labels emails as spam or not spam
// Reference for the portfolio build; it is NOT uploaded to the Space (the publish script only sends .py, .html, Dockerfile, requirements and README)
export const GPT2_SPACE_URL = "https://karamam-gpt2-demos.hf.space"; // public app address of the Space
const WAKE_TIMEOUT_MILLISECONDS = 150000; // a sleeping free Space needs about a minute to wake, so wait up to 2.5 minutes
const WAKE_POLL_MILLISECONDS = 4000; // gap between wake checks
const REQUEST_TIMEOUT_MILLISECONDS = 60000; // one answer takes 3 to 16 seconds when the line is short

function pause(milliseconds) { // resolves after the given time
  return new Promise((resolve) => setTimeout(resolve, milliseconds)); // timer wrapped in a promise
}

export async function waitForModelsReady(onStatusChange = () => {}) { // resolves when both models are loaded, rejects after the timeout
  const startedAt = Date.now(); // start of the wait
  while (Date.now() - startedAt < WAKE_TIMEOUT_MILLISECONDS) { // keep checking until the timeout
    try { // a waking Space answers with errors or no CORS headers, which fetch reports as a failure
      const reply = await fetch(GPT2_SPACE_URL + "/health", { signal: AbortSignal.timeout(15000) }); // status call
      const health = reply.ok ? await reply.json() : null; // parsed status or nothing
      if (health && health.models.includes("got") && health.models.includes("spam")) { onStatusChange("ready"); return health; } // both models loaded
    } catch (problem) { /* still waking up, try again */ }
    onStatusChange("waking"); // tell the page to show "waking up"
    await pause(WAKE_POLL_MILLISECONDS); // wait before the next check
  }
  throw new Error("The model did not wake up in time"); // give up
}

async function postGenerate(requestBody) { // one call to /generate, with a single retry when the line is full
  for (let attempt = 0; attempt < 2; attempt += 1) { // first try plus one retry
    const reply = await fetch(GPT2_SPACE_URL + "/generate", { // the shared endpoint of both demos
      method: "POST", headers: { "Content-Type": "application/json" }, // JSON body
      body: JSON.stringify(requestBody), signal: AbortSignal.timeout(REQUEST_TIMEOUT_MILLISECONDS), // stop waiting after a minute
    });
    if (reply.ok) return (await reply.json()).completion; // the answer text or the spam label
    if ((reply.status === 503 || reply.status === 429) && attempt === 0) { // line full or asking too fast
      const waitSeconds = Number(reply.headers.get("Retry-After")) || 15; // how long the Space asks us to wait
      await pause(waitSeconds * 1000); // wait, then retry once
      continue; // second attempt
    }
    throw new Error("HTTP " + reply.status); // any other failure goes to the caller
  }
}

export async function askGameOfThronesModel(question) { // returns { answer, seconds } for one question (500 characters at most)
  const startedAt = performance.now(); // start the clock
  const answer = await postGenerate({ prompt: question, task_name: "got", max_new_tokens: 120 }); // the model answers on its own and stops
  return { answer, seconds: (performance.now() - startedAt) / 1000 }; // answer and how long it took
}

export async function classifyEmailAsSpam(emailText) { // returns "spam" or "not spam" (it reads the first 120 tokens)
  return postGenerate({ prompt: emailText, task_name: "spam" }); // the classifier answers in under a second
}
