from app.agent.mappers import build_movie_search_params
from app.agent.routing import route_after_parse
from app.agent.schemas import EntityFilter, Intent, MovieQueryPlan, MovieReference
from app.db.mappers import build_movie_filters


def test_all_entity_filters_survive_mapping_without_mutating_plan():
    plan = MovieQueryPlan(
        intent=Intent.RECOMMEND_MOVIES,
        **{name: EntityFilter(include=[" A  B ", "a b", " "], exclude=[" C "])
           for name in ("genres", "actors", "directors")},
    )
    filters = build_movie_filters(build_movie_search_params({"query_plan": plan}))
    for name in ("genres", "actors", "directors"):
        assert getattr(filters, f"included_{name}") == ["a b"]
        assert getattr(filters, f"excluded_{name}") == ["c"]
        assert getattr(plan, name).include == [" A  B ", "a b", " "]
    assert route_after_parse({"query_plan": plan}) == "search_movies"
    plan.reference_movies = [MovieReference(query="Dune", exact_title=True)]
    assert route_after_parse({"query_plan": plan}) == "resolve_references"
