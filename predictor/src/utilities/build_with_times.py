from datetime import datetime, timedelta, timezone
from time import time
from typing import Optional, Iterable, Tuple

from core.dynamic_flow import DynamicFlow
from core.flow_builder import FlowBuilder


def build_with_times(
        flow_builder: FlowBuilder, flow_id: int, reroute_interval: float, horizon: float,
        observe_commodities: Optional[Iterable[int]] = None, suppress_log: bool = False
) -> Tuple[DynamicFlow, float]:
    generator = flow_builder.build_flow()
    start_time = last_milestone_time = time()
    flow = next(generator)
    start_date_time = (
            datetime(1970, 1, 1) +
            timedelta(seconds=round(start_time))
    ).replace(tzinfo=timezone.utc).astimezone(tz=None).time()
    if not suppress_log:
        print(f"Flow#{flow_id} built until phi={flow.phi}; Started At={start_date_time}")
        last_log_time = time()
    milestone_interval = reroute_interval
    milestone = milestone_interval
    while flow.phi < horizon:
        flow = next(generator)
        if flow.phi >= milestone:
            new_milestone_time = time()
            elapsed = new_milestone_time - start_time
            remaining_time = (horizon - flow.phi) * (new_milestone_time - last_milestone_time) / milestone_interval
            finish_time = (
                    datetime(1970, 1, 1) +
                    timedelta(seconds=round(new_milestone_time + remaining_time))
            ).replace(tzinfo=timezone.utc).astimezone(tz=None).time()
            if not suppress_log and new_milestone_time - last_log_time >= 1:
                print(f"Flow#{flow_id} built until phi={flow.phi:.1f};" +
                      f" Time Elapsed={timedelta(seconds=round(elapsed))};" +
                      f" Estimated Remaining Time={timedelta(seconds=round(remaining_time))};" +
                      f" Finished at {finish_time};" +
                      (
                          f" TravelTimes={[flow.avg_travel_time(i, flow.phi) for i in observe_commodities]};"
                          if observe_commodities is not None else "")
                      )
                last_log_time = new_milestone_time
            milestone += milestone_interval
            last_milestone_time = new_milestone_time
    elapsed = time() - start_time
    return flow, elapsed
