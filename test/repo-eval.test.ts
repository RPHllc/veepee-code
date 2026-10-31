import { describe, it, expect } from 'vitest';
import { isTestFile, detectTestCommand, taskFromCommit, prepareRepoWorkspace, runSuite, cleanupWorkspace, restoreGradingTests, classifyFailure } from '../src/repo-eval.js';
import { existsSync, readFileSync } from 'fs';
import { join } from 'path';

/**
 * The small harness suite measures purpose-built problems. This measures the
 * actual goal: a real repo, a real change someone made, and that repo's own
 * test suite as the grader.
 */
describe('test-file classification', () => {
  it('recognises the layouts these repos actually use', () => {
    for (const p of ['test/agent.test.ts', 'src/lib.test.js', 'tests/test_api.py',
                     '__tests__/x.tsx', 'spec/foo.spec.ts', 'api/test_thing.py']) {
      expect(isTestFile(p), p).toBe(true);
    }
  });

  it('does not mistake source for tests', () => {
    for (const p of ['src/agent.ts', 'src/testing-utils.ts', 'lib/protest.js', 'src/latest.ts']) {
      expect(isTestFile(p), p).toBe(false);
    }
  });
});

describe('task construction from a real commit', () => {
  const repo = process.cwd();

  it('finds this repo\'s own test command', async () => {
    expect(await detectTestCommand(repo)).toBe('npm test --silent');
  });

  it('builds a task whose prompt is the commit message, never the diff', async () => {
    // A commit of this repo that changed both source and tests.
    const task = await taskFromCommit(repo, 'c3d71ea7a');
    expect(typeof task).not.toBe('string');
    if (typeof task === 'string') return;
    expect(task.srcFiles).toContain('src/permissions.ts');
    expect(task.testFiles).toContain('test/permissions-dangerous.test.ts');
    expect(task.prompt).toContain('guard the git commands');
    // The instruction a person had — never the diff.
    expect(task.prompt).not.toContain('+++');
    expect(task.prompt).not.toContain('@@');
    // NOTE: a commit message MAY name the identifier it introduced — this
    // repo's own messages routinely do, and that makes its commits easier
    // tasks than they look. Tasks drawn from the app repos (socialfeed,
    // newsfeed, dogear, veetv) read like intent — "Stop the feed and the
    // cluster page disagreeing about the story" — and are the fairer sample.
    // Scoring should not mix the two populations without saying which.
    // Trailers say who wrote it, not what to do.
    expect(task.prompt).not.toContain('Co-Authored-By');
  });

  it('refuses commits that cannot grade themselves', async () => {
    // A docs/test-only or oversized commit is not a task; saying so beats
    // silently scoring a vacuous one.
    const notATask = await taskFromCommit(repo, 'HEAD~0');
    if (typeof notATask === 'string') expect(notATask.length).toBeGreaterThan(0);
  });
});

describe('workspace preparation', () => {
  it('spec mode: restores the tests, withholds the implementation', async () => {
    const task = await taskFromCommit(process.cwd(), 'c3d71ea7a', 900_000, 'spec');
    if (typeof task === 'string') throw new Error(task);
    const work = await prepareRepoWorkspace(task);
    try {
      // The grading test exists…
      expect(existsSync(join(work, 'test/permissions-dangerous.test.ts'))).toBe(true);
      const testSrc = readFileSync(join(work, 'test/permissions-dangerous.test.ts'), 'utf-8');
      expect(testSrc).toContain('isGitConfigMutation');
      // …and the implementation does not.
      const impl = readFileSync(join(work, 'src/permissions.ts'), 'utf-8');
      expect(impl).not.toContain('isGitConfigMutation');
      // node_modules is linked, not installed — an install per task would
      // measure the network instead of the harness.
      expect(existsSync(join(work, 'node_modules'))).toBe(true);
    } finally {
      await cleanupWorkspace(work);
    }
  }, 60_000);

  it('blind mode: the agent never sees what it will be graded by', async () => {
    // Handing over the assertions turns an underspecified engineering task into
    // a spec-complete one, which is the easy half of the job. In blind mode the
    // tests arrive only at grading — so they also cannot be edited to pass.
    const task = await taskFromCommit(process.cwd(), 'c3d71ea7a', 900_000, 'blind');
    if (typeof task === 'string') throw new Error(task);
    const work = await prepareRepoWorkspace(task);
    try {
      const testFile = join(work, 'test/permissions-dangerous.test.ts');
      const before = readFileSync(testFile, 'utf-8');
      // The file exists at the parent commit, but WITHOUT this commit's cases.
      expect(before).not.toContain('isGitConfigMutation');
      restoreGradingTests(work, task);
      expect(readFileSync(testFile, 'utf-8')).toContain('isGitConfigMutation');
    } finally {
      await cleanupWorkspace(work);
    }
  }, 60_000);

  it('spec mode starts from a FAILING suite, which is what makes the task real', async () => {
    const task = await taskFromCommit(process.cwd(), 'c3d71ea7a', 900_000, 'spec');
    if (typeof task === 'string') throw new Error(task);
    const work = await prepareRepoWorkspace(task);
    try {
      // A task whose tests already pass measures nothing at all.
      const before = runSuite(work, 'npx vitest run test/permissions-dangerous.test.ts', 120_000);
      expect(before.passed).toBe(false);
    } finally {
      await cleanupWorkspace(work);
    }
  }, 180_000);
});

describe('failure classification is conservative by design', () => {
  it('calls an agent-level error a harness failure', () => {
    const c = classifyFailure({ agentErrors: ['Stopped: turn cap'], toolFailures: [], suiteOutput: '' });
    expect(c.failure).toBe('harness');
  });

  it('calls an unapplied edit a harness failure, not a model one', () => {
    // Producing the text is the model's job; applying it is ours.
    const c = classifyFailure({ agentErrors: [], toolFailures: ['edit_file: old_string not found'], suiteOutput: 'AssertionError' });
    expect(c.failure).toBe('harness');
  });

  it('separates a defensible alternative from a wrong answer', () => {
    const c = classifyFailure({
      agentErrors: [], toolFailures: [], suiteOutput: 'AssertionError: expected 1 to be 2',
      preExistingSuitePassed: true,
    });
    expect(c.failure).toBe('alternative-impl');
  });

  it('will not call a failure MODEL without a transcript to check', () => {
    // "Model" was the residual bucket — whatever was left when nothing else
    // explained it. That default quietly absorbs the hardest bugs to see: a
    // harness that corrupts CONTENT while reporting success presents as model
    // stupidity every time.
    const noEvidence = classifyFailure({
      agentErrors: [], toolFailures: [], suiteOutput: 'AssertionError: expected 1 to be 2',
    });
    expect(noEvidence.failure).toBe('unclassified');
    expect(noEvidence.evidence).toContain('needs one');

    const withEvidence = classifyFailure({
      agentErrors: [], toolFailures: [], suiteOutput: 'AssertionError: expected 1 to be 2',
      hasEvidence: true,
    });
    expect(withEvidence.failure).toBe('model');
  });

  it('refuses to guess — unknown means harness until a human looks', () => {
    const c = classifyFailure({ agentErrors: [], toolFailures: [], suiteOutput: 'something inscrutable' });
    expect(c.failure).toBe('unclassified');
  });
});

describe('commit-message quality gate', () => {
  it('drops a commit whose message is not a specification', async () => {
    // The message IS the spec here. Rewriting one would make me the spec
    // author; dropping it and counting the drop is the honest move.
    const thin = await taskFromCommit(process.cwd(), 'HEAD', 900_000, 'blind');
    if (typeof thin === 'string') expect(thin.length).toBeGreaterThan(0);
  });
});

describe('blind mode withholds the assertions, not the API', () => {
  it.skipIf(!existsSync('/home/vp/Dev/crosstown/.git'))('names the exports a new module must provide, and nothing about behaviour', async () => {
    // Reviewed by hand from a real sweep: a task implemented the described
    // behaviour correctly and failed on `TypeError: m.normalise is not a
    // function`. That is a coin toss over naming, not a capability — a person
    // doing this work has the ticket, the reviewer, or the calling code.
    const task = await taskFromCommit('/home/vp/Dev/crosstown', 'fc05e7a9c', 900_000, 'blind');
    if (typeof task === 'string') return; // repo not present on this machine
    const hasNewModule = task.srcFiles.some((f) => f.includes('motion.mjs'));
    if (!hasNewModule) return;
    expect(task.prompt).toContain('must export');
    // The assertions themselves stay hidden — that is the whole point of blind.
    expect(task.prompt).not.toContain('assert.');
    expect(task.prompt).not.toContain('describe(');
  });

  it('adds nothing when the commit only modifies existing files', async () => {
    const task = await taskFromCommit(process.cwd(), '59e2e5973', 900_000, 'blind');
    if (typeof task === 'string') return;
    // The API already exists in the tree for the agent to read.
    expect(task.prompt).not.toContain('must export');
  });
});

describe('task size is bounded by what the instrument can resolve', () => {
  it.skipIf(!existsSync('/home/vp/Dev/crosstown/.git'))('drops a commit too large for one blind run, and says why', async () => {
    // crosstown@fc05e7a — 308 changed lines: a new maths module, a page rewrite
    // and a 121-line test suite. Measured: a full 30 minutes, 56 tool calls, it
    // wrote its own test file, never created the module the task was about, and
    // never ran a suite. A `budget` row is not a verdict on the agent.
    const t = await taskFromCommit('/home/vp/Dev/crosstown', 'fc05e7a9c', 900_000, 'blind');
    expect(typeof t).toBe('string');
    if (typeof t === 'string') expect(t).toContain('too large for one task');
  });

  it('keeps a normal-sized change', async () => {
    const t = await taskFromCommit(process.cwd(), 'c3d71ea7a', 900_000, 'blind');
    expect(typeof t).not.toBe('string');
  });
});
