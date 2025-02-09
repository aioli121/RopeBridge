#include <cassert>
#include <climits>
#include <cstddef>
#include <format>
#include <map>
#include <sstream>
#include <stdexcept>
#include <vector>

using time_to_cross_type = int;

struct bridge_state_type {
  static bridge_state_type start(std::size_t const people_count) {
    validate_people_count(people_count);
    return {.state_repr = one_as_int_value_type << (people_count + 1)};
  }

  static bridge_state_type end(std::size_t const people_count) {
    validate_people_count(people_count);
    return {.state_repr = (one_as_int_value_type << (people_count + 2)) - 1};
  }

  static bridge_state_type after_single_crossing(
    bridge_state_type const &prior_state,
    std::size_t const crosser_index
  ) {
    validate_crosser_index(prior_state.state_repr, crosser_index);
    return {
      .state_repr = prior_state.state_repr
        ^ torch_bit
        ^ 1 << crosser_index + 1
    };
  }

  static bridge_state_type after_double_crossing(
    bridge_state_type const &prior_state,
    std::size_t const first_crosser_index,
    std::size_t const second_crosser_index
    ) {
    validate_crosser_index(prior_state.state_repr, first_crosser_index);
    validate_crosser_index(prior_state.state_repr, second_crosser_index);
    if (first_crosser_index == second_crosser_index) {
      throw std::invalid_argument(std::format(
        "first and second crosser indices are out of range. both are {}. should be distinct.",
        first_crosser_index
      ));
    }
    return {
      .state_repr = prior_state.state_repr
        ^ torch_bit
        ^ 1 << first_crosser_index + 1
        ^ 1 << second_crosser_index + 1
    };
  }

  using int_value_type = unsigned int;
  static auto constexpr int_value_type_bit_count = sizeof(int_value_type) * CHAR_BIT;

  [[nodiscard]] int_value_type get_possible_crosser_indices() const {
    auto const torch_crossed = get_torch_crossed();
    auto result = state_repr >> 1;
    auto const people_mask = get_leading_one(result) - 1;
    result &= people_mask;
    if (!torch_crossed) {
      result ^= people_mask;
    }

    return result;
  }

  static auto constexpr min_people = 1;
  static auto constexpr max_people = int_value_type_bit_count - 2;

  //  state_repr has the bitwise form 00…001pp…ppt
  //  unused bits are on the high end and have the form 00..001 or in a special case just a leading 1
  //  p bits represent whether a given person has crossed the bridge, 0: before the bridge, 1: after
  //    the expected index of a given person is offset by 1 due to the torch bit
  //  the t bit represents the side of the torch
  int_value_type state_repr;

  struct crossing_type {
    std::size_t state_index_after_crossing;
    time_to_cross_type time_to_cross;
  };
  std::vector<crossing_type> possible_crossings;

  private:
  static void validate_people_count(std::size_t const people_count) {
    if (people_count < min_people || people_count > max_people) {
      throw std::invalid_argument(std::format(
        "people_count is out of range. is {}. should be in range [{}, {}].",
        people_count, min_people, max_people
      ));
    }
  }

  static void validate_crosser_index(int_value_type const state_repr, std::size_t const crosser_index) {
    if (
      auto const leading_one_pos = get_leading_one_pos(state_repr);
      crosser_index >= leading_one_pos - 1
    ) {
      throw std::invalid_argument(std::format(
        "crosser_index is out of range. is {}. should be in range [{}, {}].",
        crosser_index, 0, leading_one_pos - 1
      ));
    }
  }

  [[nodiscard]] bool get_torch_crossed() const {
    return (state_repr & 1) == 1;
  }

  static int_value_type get_leading_one(int_value_type const state_repr) {
    return one_as_int_value_type << get_leading_one_pos(state_repr);
  }

  static std::size_t get_leading_one_pos(int_value_type const state_repr) {
    return int_value_type_bit_count - 1 - __builtin_clz(state_repr);
  }

  static auto constexpr one_as_int_value_type = static_cast<int_value_type>(1);
  static auto constexpr torch_bit = one_as_int_value_type;

  friend std::string as_bits(bridge_state_type const &state);
};

std::string as_bits(bridge_state_type const &state) {
  auto const leading_one_pos = bridge_state_type::get_leading_one_pos(state.state_repr);

  std::ostringstream result;
  for (auto bit = bridge_state_type::one_as_int_value_type << leading_one_pos - 1; bit != 1; bit >>= 1) {
    result << (state.state_repr & bit? "1" : "0");
  }
  result << " ";
  result << (state.state_repr & 1? "1" : "0");

  return result.str();
}

using states_list_type = std::vector<bridge_state_type>;
using state_to_index_map_type = std::map<bridge_state_type::int_value_type, std::size_t>;

void try_add_or_connect_crossed_state(
  states_list_type &states,
  state_to_index_map_type &state_to_states_index,
  std::size_t &connection_count,
  std::size_t const curr_state_index,
  bridge_state_type &&crossed_state,
  time_to_cross_type const time_to_cross
) {
  auto create_connection = false;
  auto const crossed_state_index_iter = state_to_states_index.find(crossed_state.state_repr);
  std::size_t crossed_state_index;

  if (crossed_state_index_iter == state_to_states_index.end()) {
    create_connection = true;
    crossed_state_index = states.size();
    state_to_states_index.insert_or_assign(crossed_state.state_repr, crossed_state_index);
    states.emplace_back(std::move(crossed_state));
  } else if (crossed_state_index_iter->second > curr_state_index) {
    create_connection = true;
    crossed_state_index = crossed_state_index_iter->second;
  }

  if (!create_connection) {
    return;
  }

  ++connection_count;
  states.at(curr_state_index).possible_crossings.emplace_back(
    bridge_state_type::crossing_type {
      .state_index_after_crossing = crossed_state_index,
      .time_to_cross = time_to_cross
    }
  );
  states.at(crossed_state_index).possible_crossings.emplace_back(
    bridge_state_type::crossing_type {
      .state_index_after_crossing = curr_state_index,
      .time_to_cross = time_to_cross
    }
  );
}

int main() {
  std::vector<time_to_cross_type> const times_to_cross = {1,10,100,1000};

  auto const people_count = times_to_cross.size();
  auto const max_possible_states = (1 << people_count + 1) - 2;

  states_list_type states;
  states.reserve(max_possible_states);

  {
    state_to_index_map_type state_to_states_index;

    {
      auto start_state = bridge_state_type::start(people_count);
      state_to_states_index.insert_or_assign(start_state.state_repr, states.size());
      states.emplace_back(std::move(start_state));

      auto end_state = bridge_state_type::end(people_count);
      state_to_states_index.insert_or_assign(end_state.state_repr, states.size());
      states.emplace_back(std::move(end_state));
    }

    std::size_t connection_count = 0;

    for (
      decltype(states)::size_type curr_state_index = 0;
      curr_state_index < states.size();
      ++curr_state_index
    ) {
      auto const curr_state_copy = states.at(curr_state_index);
      auto const possible_crosser_indices =  curr_state_copy.get_possible_crosser_indices();

      // iterate single crosser
      {
        std::size_t single_crosser_index = 0;

        for (
          auto iterated_possible_crosser_indices = possible_crosser_indices;
          iterated_possible_crosser_indices != 0;
          ++single_crosser_index, iterated_possible_crosser_indices >>= 1
        ) {
          if ((iterated_possible_crosser_indices & 1) == 0) {
            continue;
          }

          assert(single_crosser_index < people_count);

          try_add_or_connect_crossed_state(
            states,
            state_to_states_index,
            connection_count,
            curr_state_index,
            bridge_state_type::after_single_crossing(curr_state_copy, single_crosser_index),
            times_to_cross.at(single_crosser_index)
          );
        }
      }

      // iterate double crossers
      {
        std::size_t first_crosser_index = 0;

        for (
          auto first_crosser_iterated_possible_crosser_indices = possible_crosser_indices;
          first_crosser_iterated_possible_crosser_indices != 0;
          ++first_crosser_index, first_crosser_iterated_possible_crosser_indices >>= 1
        ) {
          if ((first_crosser_iterated_possible_crosser_indices & 1) == 0) {
            continue;
          }

          assert(first_crosser_index < people_count);

          auto second_crosser_index = first_crosser_index + 1;

          for (
            auto second_crosser_iterated_possible_crosser_indices
              = first_crosser_iterated_possible_crosser_indices >> 1;
            second_crosser_iterated_possible_crosser_indices != 0;
            ++second_crosser_index, second_crosser_iterated_possible_crosser_indices >>= 1
          ) {
            if ((second_crosser_iterated_possible_crosser_indices & 1) == 0) {
              continue;
            }

            assert(second_crosser_index < people_count);

            try_add_or_connect_crossed_state(
              states,
              state_to_states_index,
              connection_count,
              curr_state_index,
              bridge_state_type::after_double_crossing(
                curr_state_copy, first_crosser_index, second_crosser_index
              ),
              std::max(
                times_to_cross.at(first_crosser_index),
                times_to_cross.at(second_crosser_index)
              )
            );
          }
        }
      }
    }
  }

  return 0;
}

/** Graph
  * ([10, 100, 1000], >, [1]), ([1, 100, 1000], >, [10]), ([1, 10, 1000], >, [100]), ([1, 10, 100], >, [1000])
  *                   |                         |                         |                         |
  *                   |1                        |10                       |100                      |1000
  *                   +--------+----------------+-------------------------+-------------------------+
  *                            |
  * START ([1, 10, 100, 1000], <, [])
  *                            |
  *                10          |             100                       1000                       100                       1000                      1000
  *               +------------+------------+-------------------------+--------------------------+-------------------------+-------------------------+
  *               |                         |                         |                          |                         |                         |
  * ([100, 1000], >, [1, 10]), ([10, 1000], >, [1, 100]), ([10, 100], >, [1, 1000]), ([1, 1000], >, [10, 100]), ([1, 100], >, [10, 1000]), ([1, 10], >, [100, 1000])
  *               |                         |                         |                          |                         |                         |
  *               |    10                   |    1                    |                          |                         |                         |
  *               +---+---------------------|---+                     |                          |                         |                         |
  *                   |                     |   |                     |                          |                         |                         |
  *                   |100                  |   |                     |    1                     |                         |                         |
  *                   +---------------------+---|---------------------|---+                      |                         |                         |
  *                   |                         |                     |   |                      |                         |                         |
  *                   |1000                     |                     |   |                      |   1                     |                         |
  *                   +-------------------------|---------------------+---|----------------------|--+                      |                         |
  *                   |                         |                         |                      |  |                      |                         |
  *                   |                         |100                      |10                    |  |                      |                         |
  *                   |                         +-------------------------+----------------------+  |                      |                         |
  *                   |                         |                         |                         |                      |                         |
  *                   |                         |1000                     |                         |10                    |                         |
  *                   |                         +-------------------------|-------------------------+----------------------+                         |
  *                   |                         |                         |                         |                                                |
  *                   |                         |                         |1000                     |100                                             |
  *                   |                         |                         +-------------------------+------------------------------------------------+
  *                   |                         |                         |                         |
  * ([10, 100, 1000], <, [1]), ([1, 100, 1000], <, [10]), ([1, 10, 1000], <, [100]), ([1, 10, 100], <, [1000])
  *                   |                         |                         |                         |
  *           100     |                 1000    |                 1000    |                         |
  *          +--------+----------------+--------|----------------+        |                         |
  *          |                         |        |                |        |                         |
  *          |100                      |1000    |                |        |                 1000    |
  *          +-------------------------+--------+----------------|--------|----------------+        |
  *          |                         |                         |        |                |        |
  *          |10                       |                         |1000    |                |1000    |
  *          +-------------------------|-------------------------+--------+----------------+        |
  *          |                         |                         |                         |        |
  *          |                         |10                       |100                      |100     |
  *          |                         +-------------------------+-------------------------+--------+
  *          |                         |                         |                         |
  * ([1000], >, [1, 10, 100]), ([100], >, [1, 10, 1000]), ([10], >, [1, 100, 1000]), ([1], >, [10, 100, 1000])
  *          |                         |                         |                         |
  *          |     100                 |     10                  |                         |      1
  *          +----+--------------------|----+--------------------|-------------------------|-----+
  *               |                    |    |                    |                         |     |
  *               |1000                |    |                    |     10                  |     |                          1
  *               +--------------------+----|--------------------|----+--------------------|-----|-------------------------+
  *               |                         |                    |    |                    |     |                         |
  *               |                         |1000                |    |100                 |     |                         |                          1
  *               |                         +--------------------+----+--------------------|-----|-------------------------|-------------------------+
  *               |                         |                         |                    |     |                         |                         |
  *               |                         |                         |                    |     |1000                     |100                      |10
  *               |                         |                         |                    +-----+-------------------------+-------------------------+
  *               |                         |                         |                          |                         |                         |
  * ([100, 1000], <, [1, 10]), ([10, 1000], <, [1, 100]), ([10, 100], <, [1, 1000]), ([1, 1000], <, [10, 100]), ([1, 100], <, [10, 1000]), ([1, 10], <, [100, 1000])
  *               |                         |                         |                          |                         |                         |
  *               |1000                     |1000                     |100                       |1000                     |100                      |10
  *          +----+-------------------------+-------------------------+--------------------------+-------------------------+-------------------------+
  *          |
  * END ([], >, [1, 10, 100, 1000])
  *          |
  *          |1000                      100                       10                        1
  *          +-------------------------+-------------------------+-------------------------+
  *          |                         |                         |                         |
  * ([1000], <, [1, 10, 100]), ([100], <, [1, 10, 1000]), ([10], <, [1, 100, 1000]), ([1], <, [10, 100, 1000])
  **/
