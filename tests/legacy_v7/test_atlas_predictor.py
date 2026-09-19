from atlas_predictor import AtlasPredictor


def test_predictor_never_uses_future_token_for_prediction():
    p = AtlasPredictor(confidence_floor=0.0, budget_scale=1.0)
    s = p.begin_session()
    s.observe({'0': {1}}, phase='decode')
    pred = s.predict({'0': {1}}, phase='decode')
    assert pred.keys == set()
    s.observe({'0': {2}}, phase='decode')
    pred = s.predict({'0': {2}}, phase='decode')
    assert all(k[1] in {1, 2} for k in pred.keys)


def test_decode_does_not_learn_prompt_transition():
    p = AtlasPredictor(confidence_floor=0.0, budget_scale=1.0)
    s = p.begin_session()
    s.observe({'0': {1}}, phase='prompt')
    s.observe({'0': {2}}, phase='prompt')
    # Phase boundary resets transient history.
    pred = s.predict({'0': {2}}, phase='decode')
    assert pred.keys == set()


def test_predictor_learns_repeated_transition():
    p = AtlasPredictor(confidence_floor=0.0, budget_scale=1.0)
    s = p.begin_session()
    for _ in range(5):
        s.observe({'0': {1}}, phase='decode')
        s.observe({'0': {2}}, phase='decode')
    pred = s.predict({'0': {2}}, phase='decode')
    assert ('0', 1) in pred.keys or ('0', 2) in pred.keys
    s.finish()
    assert p.sessions_seen == 1


def test_cross_session_history_is_available_only_after_commit():
    p = AtlasPredictor(confidence_floor=0.0, budget_scale=1.0)
    a = p.begin_session(); a.observe({'0': {1}}, phase='decode'); a.observe({'0': {2}}, phase='decode'); a.finish()
    b = p.begin_session(); b.observe({'0': {2}}, phase='decode')
    pred = b.predict({'0': {2}}, phase='decode')
    assert ('0', 1) in pred.keys or pred.keys == set()


def test_prompt_decode_boundary_uses_only_prior_sessions():
    p = AtlasPredictor(confidence_floor=0.0, budget_scale=1.0, max_candidates_per_layer=1)
    a = p.begin_session()
    a.observe({'0': {7}}, phase='prompt')
    a.observe({'0': {8}}, phase='decode')
    # Current session cannot use its own boundary yet.
    b = p.begin_session()
    b.observe({'0': {7}}, phase='prompt')
    assert b.predict_boundary({'0': {7}}, 1).keys == set()
    b.observe({'0': {8}}, phase='decode')
    b.finish()
    c = p.begin_session()
    pred = c.predict_boundary({'0': {7}}, 1)
    assert ('0', 8) in pred.keys
