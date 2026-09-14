const test = require("node:test");
const assert = require("node:assert/strict");

const {
  createManualSyncGate,
  interpolateHeading,
  parseTrajectory,
  resolveEpisodePayload,
  shortestHeadingDelta,
} = require("../tools/pano_viewer/web/trajectory.js");

function graphFixture() {
  return {
    nodes: [
      { id: "A", room_id: "Room 1" },
      { id: "B", room_id: "Room 2" },
      { id: "C", room_id: "Room 3" },
    ],
    edges: [
      { source: "A", target: "B", heading: 12 },
      { source: "B", target: "C", heading: 359 },
    ],
  };
}

function episodeFixture() {
  return {
    start_pano_id: "A",
    final_pano_id: "C",
    target_room_id: "Room 3",
    pano_path: ["A", "B", "C"],
    waypoint_room_ids: ["Room 2"],
    completed_waypoints: ["Room 2"],
    ordered_targets: ["Room 2", "Room 3"],
    success: true,
    rounds: [
      {
        round_index: 0,
        start_pano_id: "A",
        active_target_room_id: "Room 2",
        subgoal_room_id: "Room 2",
        completed_waypoints: [],
        movement_steps: [
          {
            current_pano_id: "A",
            next_pano_id: "B",
            current_room_id: "Room 1",
            next_room_id: "Room 2",
            selected_action_heading: 372,
            subgoal_room_id: "Room 2",
            active_target_room_id: "Room 2",
          },
        ],
        room_boundary: {
          from_room_id: "Room 1",
          to_room_id: "Room 2",
          at_pano_id: "B",
        },
      },
      {
        round_index: 1,
        start_pano_id: "B",
        active_target_room_id: "Room 3",
        subgoal_room_id: "Room 3",
        completed_waypoints: ["Room 2"],
        movement_steps: [
          {
            current_pano_id: "B",
            next_pano_id: "C",
            current_room_id: "Room 2",
            next_room_id: "Room 3",
          },
        ],
        room_boundary: {
          from_room_id: "Room 2",
          to_room_id: "Room 3",
          at_pano_id: "C",
        },
      },
    ],
  };
}

test("parseTrajectory expands movements, headings, and boundaries", () => {
  const trajectory = parseTrajectory(episodeFixture(), graphFixture());

  assert.deepEqual(trajectory.panoPath, ["A", "B", "C"]);
  assert.equal(trajectory.frames.length, 3);
  assert.equal(trajectory.movements.length, 2);
  assert.equal(trajectory.movements[0].actionHeading, 12);
  assert.equal(trajectory.movements[0].headingSource, "trajectory");
  assert.equal(trajectory.movements[1].actionHeading, 359);
  assert.equal(trajectory.movements[1].headingSource, "graph_edge");
  assert.equal(trajectory.frames[0].arrivalHeading, null);
  assert.equal(trajectory.frames[0].departureHeading, 12);
  assert.equal(trajectory.frames[0].heading, 12);
  assert.equal(trajectory.frames[1].arrivalHeading, 12);
  assert.equal(trajectory.frames[1].departureHeading, 359);
  assert.equal(trajectory.frames[1].heading, 12);
  assert.equal(trajectory.frames[2].arrivalHeading, 359);
  assert.equal(trajectory.frames[2].departureHeading, null);
  assert.equal(trajectory.frames[2].heading, 359);
  assert.equal(trajectory.boundaries[0].kind, "waypoint");
  assert.equal(trajectory.boundaries[1].kind, "room_transition");
});

test("heading interpolation follows the shortest circular turn", () => {
  assert.equal(shortestHeadingDelta(350, 10), 20);
  assert.equal(shortestHeadingDelta(10, 350), -20);
  assert.equal(interpolateHeading(350, 10, 0.5), 0);
  assert.equal(interpolateHeading(10, 350, 0.5), 0);
});

test("parseTrajectory accepts an evaluator wrapper", () => {
  const wrapper = {
    test_case: { test_id: "TEST001", query: "Go to Room 3" },
    episode: episodeFixture(),
    evaluation: { test_id: "TEST001", success: true },
  };

  const trajectory = parseTrajectory(wrapper, graphFixture());
  assert.equal(trajectory.sourceSchema, "evaluation_wrapper");
  assert.equal(trajectory.evaluationMetadata.testId, "TEST001");
  assert.deepEqual(trajectory.panoPath, ["A", "B", "C"]);
});

test("resolveEpisodePayload reports evaluator records without a completed episode", () => {
  assert.throws(
    () => resolveEpisodePayload({ episode: null, episode_error: "navigation failed" }),
    /does not contain a completed episode: navigation failed/,
  );
});

test("parseTrajectory rejects unknown panos and illegal graph edges", () => {
  const unknown = episodeFixture();
  unknown.pano_path[2] = "missing";
  assert.throws(() => parseTrajectory(unknown, graphFixture()), /unknown pano/);

  const illegal = episodeFixture();
  illegal.pano_path = ["A", "C", "B"];
  illegal.rounds[0].movement_steps[0].next_pano_id = "C";
  illegal.rounds[1].movement_steps[0].current_pano_id = "C";
  illegal.rounds[1].movement_steps[0].next_pano_id = "B";
  illegal.rounds[0].room_boundary.at_pano_id = "C";
  assert.throws(() => parseTrajectory(illegal, graphFixture()), /not a legal graph edge/);
});

test("manual sync gate waits for explicit Street View load", async () => {
  const events = [];
  const gate = createManualSyncGate({
    load: async () => events.push("load"),
    sync: (value) => events.push(["sync", value]),
  });

  gate.update({ panoId: "A", heading: 12 });
  gate.update({ panoId: "B", heading: 20 });
  assert.deepEqual(events, []);

  await gate.load();
  assert.deepEqual(events, ["load", ["sync", { panoId: "B", heading: 20 }]]);

  gate.update({ panoId: "C", heading: 30 });
  assert.deepEqual(events.at(-1), ["sync", { panoId: "C", heading: 30 }]);
});
