"""Behavioral tests with the real LifeMate simulator; all writes stay in /tmp."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import Pipeline, PipelineError


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        source = Path(__file__).resolve().parents[1] / "train/train_tools_catalog.json"
        (self.out / "train_tools_catalog.json").write_bytes(source.read_bytes())
        self.p = Pipeline(self.out)
        alarm = [t["name"] for t in self.p.catalog["apps"] if t["id"] == "alarm" for t in t["tools"]]
        self.draft = {
            "id": "syn_7001", "audio_text": "Sáng mai đặt báo thức sáu giờ nhé.",
            "context": {
                "current_time": "2026-11-02T20:00:00+07:00", "timezone": "Asia/Ho_Chi_Minh",
                "location": {"lat": 21.0285, "lng": 105.8542},
                "user_profile": {"name": "Mai", "city": "Hà Nội", "home_address": "12 Hàng Bông", "work_address": "8 Tràng Tiền"},
                "installed_apps": {"alarm": {"tools": alarm}},
            },
            "history": [], "initial_state": {"alarm": {"alarms": []}},
            "tool_calls": [{"name": "alarm.create", "arguments": {"time": "2026-11-03T06:00:00+07:00"}}],
            "spec": {
                "author": "writer-a", "family": "time", "intent": "Tạo báo thức sáng mai lúc sáu giờ.",
                "goals": [{"path": ["alarm", "alarms", 0, "time"], "op": "eq", "value": "2026-11-03T06:00:00+07:00"}],
            },
        }

    def stage(self):
        return self.p.stage("g7", copy.deepcopy(self.draft))

    def solve(self, calls=None):
        self.p.start_solver("syn_7001", "solver-b")
        for c in self.draft["tool_calls"] if calls is None else calls:
            self.p.solver_call("syn_7001", "solver-b", c)
        return self.p.finish_solver("syn_7001", "solver-b")

    def approve(self, sid='syn_7001'):
        report = {"fingerprint": self.p.fingerprint(self.p.sample(sid)), "verdict": "approve", "checks": {
            k: {"passed": True, "evidence": "Người dùng nói sáu giờ sáng mai, tương ứng 2026-11-03T06:00:00+07:00; một alarm.create là đủ."}
            for k in self.p.REVIEW_CHECKS
        }}
        self.p.approve(sid, "reviewer-c", report)

    def test_stage_generates_labels_without_any_tts_files(self):
        row = self.stage()
        self.assertEqual(row["label"]["tool_results"][0]["id"], "alm_01")
        self.assertEqual(self.p.validate(row)[0], [])
        self.assertFalse((self.out / "audio").exists())

    def test_rejects_bad_ask_target_and_parameter(self):
        for args in [{"tool_name": "absent.fake", "params": ["time"]}, {"tool_name": "alarm.create", "params": ["made_up"]}]:
            with self.subTest(args=args), self.assertRaises(PipelineError):
                d = copy.deepcopy(self.draft)
                d["tool_calls"] = [{"name": "sys.ask_missing_info", "arguments": args}]
                self.p.stage("g7", d)

    def test_goal_catches_omitted_action_even_when_replay_is_consistent(self):
        self.draft["tool_calls"] = []
        with self.assertRaisesRegex(PipelineError, "goal"):
            self.stage()

    def test_rejects_hidden_reference_id(self):
        self.draft["initial_state"]["alarm"]["alarms"] = [{"id": "alm_01", "time": "2026-11-03T05:00:00+07:00"}]
        self.draft["tool_calls"] = [{"name": "alarm.update", "arguments": {"alarm_id": "alm_01", "time": "2026-11-03T06:00:00+07:00"}}]
        with self.assertRaisesRegex(PipelineError, "chưa.*tra|x-ref"):
            self.stage()

    def test_rejects_literal_home_in_route(self):
        d = copy.deepcopy(self.draft)
        maps = next(a for a in self.p.catalog["apps"] if a["id"] == "maps")
        d["context"]["installed_apps"] = {"maps": {"tools": [t["name"] for t in maps["tools"]]}}
        d["initial_state"] = {"maps": {"places": [], "saved_places": []}}
        d["tool_calls"] = [{"name": "maps.get_route", "arguments": {"origin": "nhà", "destination": "Bệnh viện"}}]
        with self.assertRaisesRegex(PipelineError, "profile|địa chỉ"):
            self.p.stage("g7", d)

    def test_history_runs_once_before_final_turn(self):
        self.draft["history"] = [
            {"role": "user", "audio_text": "Đặt báo thức năm giờ sáng mai."},
            {"role": "assistant", "text": "Đã đặt báo thức năm giờ.", "tool_calls": [{"name": "alarm.create", "arguments": {"time": "2026-11-03T05:00:00+07:00"}}]},
        ]
        self.draft["spec"]["goals"][0]["path"][2] = 1
        row = self.stage()
        self.assertEqual(row["label"]["tool_results"][0]["id"], "alm_02")
        self.assertEqual(self.p.validate(row)[0], [])
        (self.out / "work/syn_7001/initial_state.json").unlink()
        self.assertTrue(self.p.validate(row)[0])

    def test_solver_packet_hides_gold_states_and_goals(self):
        self.stage()
        packet = self.p.packet("syn_7001")
        self.assertEqual(set(packet), {"sample", "tools"})
        self.assertEqual(set(packet["sample"]), {"id", "audio_text", "context", "history"})
        self.assertNotIn("label", packet["sample"])
        self.assertNotIn("initial_state", packet["sample"])

    def test_requires_distinct_author_solver_and_reviewer(self):
        self.stage()
        with self.assertRaises(PipelineError):
            self.p.start_solver("syn_7001", "writer-a")
        self.solve()
        for actor in ["writer-a", "solver-b"]:
            with self.assertRaises(PipelineError):
                self.p.approve("syn_7001", actor, {})

    def test_failed_independent_solution_cannot_be_approved(self):
        self.stage()
        result = self.solve([])
        self.assertFalse(result["agrees"])
        with self.assertRaises(PipelineError):
            self.approve()

    def test_review_requires_specific_evidence(self):
        self.stage()
        self.solve()
        with self.assertRaises(PipelineError):
            self.p.approve("syn_7001", "reviewer-c", {"verdict": "approve", "checks": {k: True for k in self.p.REVIEW_CHECKS}})

    def test_release_only_contains_finished_and_independently_approved_samples(self):
        self.stage()
        self.p.seal("g7")
        with self.assertRaises(PipelineError):
            self.p.build()
        self.solve()
        self.approve()
        release = self.p.build()
        rows = [json.loads(l) for l in Path(release["data_path"]).read_text().splitlines()]
        self.assertEqual([r["id"] for r in rows], ["syn_7001"])
        self.assertEqual(release["n"], 1)

    def test_stale_review_cannot_enter_release(self):
        self.stage()
        self.solve()
        self.approve()
        self.p.seal("g7")
        self.draft["audio_text"] = "Em đặt báo thức sáu giờ sáng mai nha."
        self.p.stage("g7", self.draft, replace=True)
        self.p.seal("g7")
        with self.assertRaises(PipelineError):
            self.p.build()

    def test_unsealed_partial_group_cannot_break_release(self):
        self.stage()
        self.solve()
        self.approve()
        self.p.seal("g7")
        (self.out / "parts/g8.jsonl").write_text('{"id":')
        self.assertEqual(self.p.build()["n"], 1)

    def test_corrupt_sealed_group_keeps_previous_release_pointer(self):
        self.stage()
        self.solve()
        self.approve()
        self.p.seal("g7")
        self.p.build()
        pointer = (self.out / "verified.json").read_bytes()
        with (self.out / "parts/g7.jsonl").open("a") as f:
            f.write('{"id":')
        with self.assertRaises(PipelineError):
            self.p.build()
        self.assertEqual((self.out / "verified.json").read_bytes(), pointer)

    def test_duplicate_ids_are_rejected(self):
        self.stage()
        with self.assertRaises(PipelineError):
            self.p.stage("g8", self.draft)

    def test_counterfactual_pair_allows_same_words_but_requires_different_behavior(self):
        self.draft["spec"]["pair"] = {"id": "p1", "dimension": "known alarm count", "variant": "one"}
        self.stage()
        other = copy.deepcopy(self.draft)
        other["id"] = "syn_7002"
        other["spec"]["pair"]["variant"] = "two"
        self.p.stage("g7", other)
        with self.assertRaisesRegex(PipelineError, "pair"):
            self.p.seal("g7")

    def test_cli_works_without_tts(self):
        self.stage()
        import os
        env = dict(os.environ, SYN_OUT=str(self.out), PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run([sys.executable, str(Path(__file__).parent / "check_sample.py"), "--all", "-q"], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_same_sentence_in_different_groups_is_rejected(self):
        self.stage()
        d = copy.deepcopy(self.draft)
        d['id'] = 'syn_8001'
        self.p.stage('g8', d)
        with self.assertRaisesRegex(PipelineError, 'trùng'):
            self.p.seal('g8')

    def test_claims_are_unique_under_concurrency(self):
        import os
        env = dict(os.environ, SYN_OUT=str(self.out), PYTHONDONTWRITEBYTECODE='1')
        cmd = [sys.executable, str(Path(__file__).parent / 'claim_group.py'), '--apps', 'alarm', '--n', '2']
        procs = [subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(4)]
        groups = []
        for proc in procs:
            stdout, stderr = proc.communicate(timeout=20)
            self.assertEqual(proc.returncode, 0, stderr)
            groups.append(json.loads(stdout)['GROUP'])
        self.assertEqual(len(set(groups)), 4)
        self.assertEqual(len(list((self.out / 'parts').glob('g*.claim.json'))), 4)

    def test_group_cannot_finish_before_claimed_count(self):
        from pipeline import atomic_json
        self.stage()
        atomic_json(self.out / 'parts/g7.claim.json', {'N': 2, 'COUNTERFACTUAL_PAIRS': 0})
        with self.assertRaisesRegex(PipelineError, 'N|số mẫu'):
            self.p.seal('g7')

    def test_seal_works_with_equals_argument(self):
        import os
        self.stage()
        env = dict(os.environ, SYN_OUT=str(self.out), PYTHONDONTWRITEBYTECODE='1')
        result = subprocess.run([sys.executable, str(Path(__file__).parent / 'claim_group.py'), '--finish=g7'], env=env, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_rejects_hidden_fields_in_blind_input(self):
        for location in ['context', 'history']:
            d = copy.deepcopy(self.draft)
            if location == 'context':
                d['context']['gold_tool_calls'] = d['tool_calls']
            else:
                d['history'] = [{'role': 'user', 'audio_text': 'Chào bạn.', 'gold_tool_calls': d['tool_calls']}, {'role': 'assistant', 'text': 'Chào bạn.'}]
            with self.subTest(location=location), self.assertRaises(PipelineError):
                self.p.stage('g7', d)

    def test_report_must_identify_the_snapshot_reviewed(self):
        self.stage()
        self.solve()
        self.approve()
        old_report = json.loads((self.out / 'work/syn_7001/review.json').read_text())['report']
        self.draft['audio_text'] = 'Sáng mai đặt báo thức bảy giờ nhé.'
        self.draft['tool_calls'][0]['arguments']['time'] = '2026-11-03T07:00:00+07:00'
        self.draft['spec']['goals'][0]['value'] = '2026-11-03T07:00:00+07:00'
        self.p.stage('g7', self.draft, replace=True)
        self.p.start_solver('syn_7001', 'solver-new')
        self.p.solver_call('syn_7001', 'solver-new', self.draft['tool_calls'][0])
        self.p.finish_solver('syn_7001', 'solver-new')
        with self.assertRaisesRegex(PipelineError, 'fingerprint|snapshot'):
            self.p.approve('syn_7001', 'reviewer-new', old_report)

    def test_pair_is_released_together_or_not_at_all(self):
        self.draft['audio_text'] = 'Nếu chưa có báo thức sáu giờ sáng mai thì đặt giúp tôi.'
        self.draft['spec']['pair'] = {'id': 'g7-pair', 'dimension': 'existing matching alarm', 'variant': 'absent'}
        self.stage()
        d = copy.deepcopy(self.draft)
        d['id'] = 'syn_7002'
        d['initial_state']['alarm']['alarms'] = [{'id': 'alm_11', 'time': '2026-11-03T06:00:00+07:00'}]
        d['tool_calls'] = []
        d['spec']['goals'] = [{'op': 'call_count', 'value': 0}]
        d['spec']['pair']['variant'] = 'present'
        self.p.stage('g7', d)
        self.p.seal('g7')
        self.solve()
        self.approve()
        with self.assertRaisesRegex(PipelineError, '0 mẫu'):
            self.p.build()
        self.p.start_solver('syn_7002', 'solver-b')
        self.p.finish_solver('syn_7002', 'solver-b')
        self.approve('syn_7002')
        self.assertEqual(self.p.build()['n'], 2)

    def test_stage_enforces_reserved_claim_range(self):
        from pipeline import atomic_json
        atomic_json(self.out / 'parts/g7.claim.json', {'N': 1, 'ID_FIRST': 'syn_7001', 'ID_LAST': 'syn_7001'})
        self.draft['id'] = 'syn_7002'
        with self.assertRaisesRegex(PipelineError, 'claim|dải'):
            self.stage()

    def test_old_release_manifest_is_immutable_after_new_drafts_appear(self):
        self.stage()
        self.solve()
        self.approve()
        self.p.seal('g7')
        first = self.p.build()
        manifest = Path(first['data_path']).with_name('manifest.json')
        snapshot = manifest.read_bytes()
        (self.out / 'parts/g8.jsonl').write_text('')
        self.p.build()
        self.assertEqual(manifest.read_bytes(), snapshot)

    def test_malformed_review_returns_domain_error(self):
        self.stage()
        self.solve()
        with self.assertRaises(PipelineError):
            self.p.approve('syn_7001', 'reviewer-c', [])

    def test_fallback_never_assigns_a_holdout_app(self):
        import claim_group
        previous = claim_group.PARTS
        claim_group.PARTS = self.out / 'parts'
        try:
            apps = claim_group.least_covered({'fitness', 'flights', 'music', 'tv', 'alarm'})
            self.assertEqual(apps, 'alarm')
        finally:
            claim_group.PARTS = previous

    def test_cli_end_to_end_publishes_a_verified_release(self):
        import os
        env = dict(os.environ, SYN_OUT=str(self.out), PYTHONDONTWRITEBYTECODE='1')
        scripts = Path(__file__).parent
        def run(script, *args):
            result = subprocess.run([sys.executable, str(scripts / script), *args], capture_output=True, text=True, env=env, timeout=20)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return result.stdout
        proposal = self.out / 'proposal.json'
        proposal.write_text(json.dumps(self.draft))
        run('stage_sample.py', '--group', 'g7', '--input', str(proposal))
        run('claim_group.py', '--finish', 'g7')
        run('review_sample.py', 'start', '--id', 'syn_7001', '--actor', 'solver-b')
        call = run('review_sample.py', 'call', '--id', 'syn_7001', '--actor', 'solver-b', '--call', json.dumps(self.draft['tool_calls'][0]))
        self.assertEqual(json.loads(call)['item']['time'], '2026-11-03T06:00:00+07:00')
        run('review_sample.py', 'finish', '--id', 'syn_7001', '--actor', 'solver-b')
        packet = json.loads(run('review_sample.py', 'review-packet', '--id', 'syn_7001', '--actor', 'reviewer-c'))
        report = self.out / 'report.json'
        report.write_text(json.dumps({'fingerprint': packet['fingerprint'], 'verdict': 'approve', 'checks': {k: {'passed': True, 'evidence': 'Fixture: user requests six tomorrow; 2026-11-03T06:00:00+07:00 and one alarm.create.'} for k in self.p.REVIEW_CHECKS}}))
        run('review_sample.py', 'approve', '--id', 'syn_7001', '--actor', 'reviewer-c', '--report', str(report))
        run('check_sample.py', '--all', '--build')
        published = Path(run('check_sample.py', '--verified-path').strip())
        self.assertEqual(json.loads(published.read_text())['id'], 'syn_7001')
        self.assertFalse((self.out / 'tts_input.jsonl').exists())

    def test_incomplete_input_attempt_can_be_excluded_without_losing_trace(self):
        self.stage()
        self.solve()
        self.approve()
        self.assertTrue(self.p.ready(self.p.sample('syn_7001')))
        self.p.exclude_solver('syn_7001', 'solver-b', 'Input packet was truncated before solver saw full history.')
        self.assertFalse(self.p.ready(self.p.sample('syn_7001')))
        attempt = json.loads(self.p._solver_path('syn_7001', 'solver-b').read_text())
        self.assertEqual(attempt['calls'], self.draft['tool_calls'])
        self.assertTrue(attempt['agrees'])
        self.assertIn('truncated', attempt['excluded_reason'])
        with self.assertRaises(PipelineError):
            self.approve()

    def test_case_and_whitespace_are_canonical_but_extra_optional_args_still_disagree(self):
        self.draft['audio_text'] = 'Đặt báo thức sáu giờ sáng mai, nhãn ghé nhà thuốc.'
        self.draft['tool_calls'][0]['arguments']['label'] = 'Ghé nhà thuốc'
        self.stage()
        calls = copy.deepcopy(self.draft['tool_calls'])
        calls[0]['arguments']['label'] = 'ghé  nhà thuốc'
        self.assertTrue(self.solve(calls)['agrees'])
        self.p.start_solver('syn_7001', 'solver-extra')
        calls[0]['arguments']['repeat'] = 'once'
        self.p.solver_call('syn_7001', 'solver-extra', calls[0])
        self.assertFalse(self.p.finish_solver('syn_7001', 'solver-extra')['agrees'])

    def test_coordinator_recheck_retains_old_verdict_without_new_solver_calls(self):
        self.stage()
        self.solve()
        path = self.p._solver_path('syn_7001', 'solver-b')
        old = json.loads(path.read_text())
        old['agrees'] = False
        path.write_text(json.dumps(old))
        result = self.p.recheck_solver('syn_7001', 'solver-b')
        self.assertTrue(result['agrees'])
        updated = json.loads(path.read_text())
        self.assertEqual(updated['calls'], self.draft['tool_calls'])
        self.assertFalse(updated['agreement_history'][0]['agrees'])


if __name__ == "__main__":
    unittest.main()
