import datetime
from typing import Optional, Iterable

import time

from core.multi_com_dynamic_flow import MultiComPartialDynamicFlow
from core.multi_com_flow_builder import MultiComFlowBuilder


def build_with_times(
        flow_builder: MultiComFlowBuilder, flow_id: int, reroute_interval: float, horizon: float,
        observe_commodities: Optional[Iterable[int]] = None, suppress_log: bool = False
) -> MultiComPartialDynamicFlow:
    generator = flow_builder.build_flow()
    start_time = last_milestone_time = time.time()
    flow = next(generator)
    start_date_time = (
            datetime.datetime(1970, 1, 1) +
            datetime.timedelta(seconds=round(start_time))
    ).replace(tzinfo=datetime.timezone.utc).astimezone(tz=None).time()
    if not suppress_log:
        print(f"Flow#{flow_id} built until phi={flow.phi}; Started At={start_date_time}")
    milestone = reroute_interval
    while flow.phi < horizon:
        flow = next(generator)
        if flow.phi >= milestone:
            new_milestone_time = time.time()
            elapsed = new_milestone_time - start_time
            remaining_time = (horizon - flow.phi) * (new_milestone_time - last_milestone_time) / reroute_interval
            finish_time = (
                    datetime.datetime(1970, 1, 1) +
                    datetime.timedelta(seconds=round(new_milestone_time + remaining_time))
            ).replace(tzinfo=datetime.timezone.utc).astimezone(tz=None).time()
            if not suppress_log:
                print(f"Flow#{flow_id} built until phi={flow.phi:.1f};" +
                      f" Time Elapsed={datetime.timedelta(seconds=round(elapsed))};" +
                      f" Estimated Remaining Time={datetime.timedelta(seconds=round(remaining_time))};" +
                      f" Finished at {finish_time};" +
                      (
                          f" TravelTimes={[flow.avg_travel_time(i, flow.phi) for i in observe_commodities]};"
                          if observe_commodities is not None else "")
                      )
            milestone += reroute_interval
            last_milestone_time = new_milestone_time

    return flow
