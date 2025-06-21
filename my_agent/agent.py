import numpy as np
import tensorflow as tf
from game.players import BasePokerPlayer
from os.path import dirname
import random
import json

class trainedPlayer(BasePokerPlayer):
    
    # Preflop win rates for all 169 possible starting hands
    preflop_win_rates = {
        'AA': 85, 'AK': 68, 'AQ': 67, 'AJ': 66, 'AT': 66, 'A9': 64, 'A8': 63, 'A7': 63, 'A6': 62, 'A5': 61, 'A4': 60, 'A3': 60, 'A2': 59,
        'KA': 66, 'KK': 83, 'KQ': 64, 'KJ': 63, 'KT': 61, 'K9': 60, 'K8': 59, 'K7': 58, 'K6': 57, 'K5': 56, 'K4': 55, 'K3': 55, 'K2': 54,
        'QA': 65, 'QK': 62, 'QQ': 80, 'QJ': 61, 'QT': 59, 'Q9': 58, 'Q8': 56, 'Q7': 54, 'Q6': 53, 'Q5': 52, 'Q4': 51, 'Q3': 50, 'Q2': 49,
        'JA': 65, 'JK': 62, 'JQ': 62, 'JJ': 78, 'JT': 59, 'J9': 57, 'J8': 55, 'J7': 53, 'J6': 52, 'J5': 50, 'J4': 49, 'J3': 48, 'J2': 47,
        'TA': 64, 'TK': 61, 'TQ': 59, 'TJ': 57, 'TT': 75, 'T9': 54, 'T8': 53, 'T7': 51, 'T6': 49, 'T5': 48, 'T4': 46, 'T3': 45, 'T2': 44,
        '9A': 62, '9K': 59, '9Q': 58, '9J': 56, '9T': 54, '99': 72, '98': 53, '97': 51, '96': 50, '95': 48, '94': 47, '93': 46, '92': 45,
        '8A': 61, '8K': 58, '8Q': 56, '8J': 55, '8T': 53, '89': 52, '88': 69, '87': 50, '86': 49, '85': 47, '84': 46, '83': 45, '82': 43,
        '7A': 60, '7K': 57, '7Q': 54, '7J': 53, '7T': 51, '79': 50, '78': 49, '77': 67, '76': 48, '75': 47, '74': 45, '73': 44, '72': 43,
        '6A': 59, '6K': 56, '6Q': 53, '6J': 52, '6T': 50, '69': 48, '68': 47, '67': 46, '66': 64, '65': 46, '64': 45, '63': 44, '62': 42,
        '5A': 59, '5K': 56, '5Q': 52, '5J': 50, '5T': 48, '59': 47, '58': 45, '57': 46, '56': 45, '55': 61, '54': 44, '53': 43, '52': 41,
        '4A': 58, '4K': 55, '4Q': 51, '4J': 49, '4T': 47, '49': 45, '48': 44, '47': 44, '46': 44, '45': 43, '44': 58, '43': 42, '42': 40,
        '3A': 58, '3K': 55, '3Q': 50, '3J': 48, '3T': 46,'39': 44, '38': 43, '37': 43, '36': 43, '35': 42, '34': 41, '33': 55, '32': 39,
        '2A': 57, '2K': 54, '2Q': 49, '2J': 47, '2T': 44, '29': 42, '28': 40, '27': 39, '26': 39, '25': 38, '24': 37, '23': 36, '22': 51
    }
    
    def __init__(self):
        super().__init__()
        # Core game state
        self.hole_cards = []
        self.community_cards = []
        self.stack = 0
        self.opponent_stack = 0
        self.remaining_rounds = 20
        self.is_p1 = False
        self.round_count = 0
        
        # Neural network component
        self.model = self.load_model('model.h5')
        
        # Betting and stack management
        self.latest_bet = {}
        self.total_bet = 0
        self.total_loss = 0
        self.flag = False  # Preflop first action flag
        
        # Hyperparameters as defined in report
        self.emergency_threshold = 1050
        self.base_bluff_frequency = 0.1
        self.premium_threshold = 70
        self.good_hand_threshold = 55
        self.marginal_threshold = 42
        self.opponent_history_window = 10
        
        # Opponent modeling system
        self.opponent_actions = []
        self.opponent_fold_rate = 0.5  # Default assumption
        self.opponent_raise_rate = 0.3  # Default assumption
        self.opponent_aggressive_count = 0
        self.opponent_total_actions = 0
        
        # Performance tracking
        self.game_history = []
        self.decision_log = []

    def load_model(self, model_path):
        # load model
        try:
            return tf.keras.models.load_model(f'{dirname(__file__)}/{model_path}')
        except Exception as e:
            print(f"Model loading failed: {e}. Using rule-based fallback.")
            return None

    def declare_action(self, valid_actions, hole_card, round_state):
        # Update game state
        self.hole_cards = hole_card
        self.community_cards = round_state['community_card']
        self.stack = round_state['seats'][round_state['next_player']]['stack']
        self.opponent_stack = round_state['seats'][1 - round_state['next_player']]['stack']
        self.remaining_rounds = 21 - round_state['round_count']
        self.is_p1 = round_state['seats'][0]['uuid'] == self.uuid
        self.round_count = round_state['round_count']

        # Emergency stack protection (Tournament Survival Strategy)
        if self.should_switch_to_folding():
            action, amount = self.find_action(valid_actions, 'fold')
            return self.safe_action(action, amount, valid_actions)

        # Decision making based on game phase
        if not self.community_cards:
            # preflop
            action, amount = self.enhanced_preflop_strategy(valid_actions)
        else:
            # post flop
            action, amount = self.hybrid_postflop_strategy(valid_actions, round_state)

        action, amount = self.safe_action(action, amount, valid_actions)
        self.log_decision(action, amount, round_state)
        
        # update betting
        self.update_betting_state(action, amount, round_state)
        
        return action, amount

    def enhanced_preflop_strategy(self, valid_actions):
        # Get hand key for win rate lookup
        sorted_hole_cards = sorted(self.hole_cards, key=lambda card: card[1])
        hand_key = ''.join(card[1] for card in sorted_hole_cards)
        if sorted_hole_cards[0][0] == sorted_hole_cards[1][0]:  # same suit
            hand_key = hand_key[::-1]

        base_win_rate = self.preflop_win_rates.get(hand_key, 40)
        
        # Calculate contextual adjustment factors
        late_game_factor = 1.2 if self.remaining_rounds <= 5 else 1.0
        stack_factor = 0.8 if self.stack < 500 else 1.0
        
        # Apply adaptive thresholds
        adjusted_win_rate = base_win_rate * late_game_factor * stack_factor
        
       # get random probability
        p = np.random.rand() * 100
        
        # decision tree
        if adjusted_win_rate >= self.premium_threshold:  # Premium hands (≥80%)
            call_prob = (100 - adjusted_win_rate) / 8
            if p < call_prob:
                return valid_actions[1]['action'], valid_actions[1]['amount']  
            else:
                # Aggressive raise
                target_amount = self.calculate_aggressive_raise(valid_actions)
                self.flag = True
                return valid_actions[2]['action'], target_amount
                
        elif adjusted_win_rate >= self.good_hand_threshold:  # Good hands (≥65%)
            call_prob = (100 - adjusted_win_rate) / 3
            if p < call_prob:
                return valid_actions[1]['action'], valid_actions[1]['amount'] 
            else:
                # Moderate raise
                target_amount = self.calculate_moderate_raise(valid_actions)
                return valid_actions[2]['action'], target_amount
                
        elif adjusted_win_rate >= self.marginal_threshold:  # Marginal hands (≥45%)
            call_prob = (100 - adjusted_win_rate) / 4
            if p < call_prob:
                return valid_actions[1]['action'], valid_actions[1]['amount']  
            else:
                return valid_actions[0]['action'], valid_actions[0]['amount']  
        else:  # Weak hands (<42%)
            # Occasional deception play
            if p < 8 and self.remaining_rounds > 10:
                return valid_actions[1]['action'], valid_actions[1]['amount']  
            else:
                return valid_actions[0]['action'], valid_actions[0]['amount']  

    def hybrid_postflop_strategy(self, valid_actions, round_state):
        # Calculate hand strength and pot context
        hand_strength = self.estimate_hand_strength()
        pot_size = self.calculate_pot_size(round_state)
        call_amount = valid_actions[1]['amount'] if len(valid_actions) > 1 else 0
        
        # Calculate pot odds
        pot_odds = pot_size / call_amount if call_amount > 0 else float('inf')
        
        # Get neural network prediction if available
        model_suggestion = None
        if self.model:
            try:
                state = self.get_enhanced_state(round_state)
                state = np.reshape(state, [1, 10])  # Ensure 10 dimensions for existing model
                q_values = self.model.predict(state, verbose=0)
                action_idx = np.argmax(q_values[0])
                model_suggestion = self.map_action_index(action_idx)
            except Exception as e:
                model_suggestion = 'call'
        
        # Hybrid decision making based on hand strength
        if hand_strength >= 0.8:  # Very strong hands
            # Value betting strategy
            if np.random.rand() < 0.9:  # 90% frequency
                bet_amount = self.calculate_value_bet(pot_size, valid_actions, 0.8)
                return self.get_raise_action(valid_actions, bet_amount)
            else:
                return valid_actions[1]['action'], valid_actions[1]['amount']  
                
        elif hand_strength >= 0.6:  # Good hands
            # Mixed strategy: moderate aggression
            if np.random.rand() < 0.6:  # 60% frequency
                bet_amount = self.calculate_value_bet(pot_size, valid_actions, 0.5)
                return self.get_raise_action(valid_actions, bet_amount)
            else:
                return valid_actions[1]['action'], valid_actions[1]['amount']  
                
        elif hand_strength >= 0.3:  # Marginal hands
            # Pot odds-based decisions
            if pot_odds >= 3 or call_amount <= self.stack * 0.1:
                return valid_actions[1]['action'], valid_actions[1]['amount'] 
            else:
                return valid_actions[0]['action'], valid_actions[0]['amount']  
        else:  # Weak hands
            # Bluffing strategy with opponent modeling
            bluff_frequency = self.calculate_adaptive_bluff_frequency()
            if np.random.rand() < bluff_frequency and pot_size < self.stack * 0.3:
                bet_amount = self.calculate_bluff_bet(pot_size, valid_actions)
                return self.get_raise_action(valid_actions, bet_amount)
            else:
                return valid_actions[0]['action'], valid_actions[0]['amount']  

    def estimate_hand_strength(self):
        if not self.community_cards:
            return 0.5  
            
        # Combine hole cards and community cards
        all_cards = self.hole_cards + self.community_cards
        ranks = [self.card_to_rank(card[1]) for card in all_cards]
        suits = [card[0] for card in all_cards]
        
        # Count rank occurrences
        rank_counts = {}
        for rank in ranks:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
            
        max_count = max(rank_counts.values()) if rank_counts else 1
        
        # Hand strength classification
        if max_count >= 4:
            return 0.95  # Four of a kind
        elif max_count >= 3:
            # Check for full house
            counts = sorted(rank_counts.values(), reverse=True)
            if len(counts) >= 2 and counts[1] >= 2:
                return 0.90  # Full house
            else:
                return 0.75  # Three of a kind
        elif max_count >= 2:
            # Check for two pair
            pair_count = sum(1 for count in rank_counts.values() if count >= 2)
            if pair_count >= 2:
                return 0.60  # Two pair
            else:
                return 0.55  # One pair
        else:
            # High card evaluation
            hole_ranks = [self.card_to_rank(card[1]) for card in self.hole_cards]
            if max(hole_ranks) >= 11:  # Jack or better
                return 0.4
            else:
                return 0.2

    def update_opponent_model(self, action):
        self.opponent_actions.append(action)
        self.opponent_total_actions += 1
        
        if action in ['raise', 'bet']:
            self.opponent_aggressive_count += 1
        
        # Update statistics using sliding window
        if len(self.opponent_actions) > self.opponent_history_window:
            recent_actions = self.opponent_actions[-self.opponent_history_window:]
            self.opponent_fold_rate = recent_actions.count('fold') / len(recent_actions)
            self.opponent_raise_rate = recent_actions.count('raise') / len(recent_actions)

    def calculate_adaptive_bluff_frequency(self):
        base_frequency = self.base_bluff_frequency
        
        # Adjust based on opponent fold rate
        if self.opponent_fold_rate > 0.6:
            return base_frequency * 1.5 
        elif self.opponent_fold_rate < 0.3:
            return base_frequency * 0.5  
        else:
            return base_frequency

    def calculate_value_bet(self, pot_size, valid_actions, multiplier):
        target_bet = pot_size * multiplier
        
        if len(valid_actions) > 2:
            raise_info = valid_actions[2]['amount']
            if isinstance(raise_info, dict):
                if raise_info['min'] == -1:
                    return valid_actions[1]['amount']  # Can't raise
                return max(min(target_bet, raise_info['max']), raise_info['min'])
            else:
                return raise_info
        return target_bet

    def calculate_bluff_bet(self, pot_size, valid_actions):
        return self.calculate_value_bet(pot_size, valid_actions, 0.6)

    def calculate_aggressive_raise(self, valid_actions):
        target_bet = 1001 - self.stack + self.total_loss
        
        if len(valid_actions) > 2:
            raise_info = valid_actions[2]['amount']
            if isinstance(raise_info, dict):
                if raise_info['min'] == -1:
                    return valid_actions[1]['amount']
                return max(min(target_bet, raise_info['max']), raise_info['min'])
            else:
                return raise_info
        return target_bet

    def calculate_moderate_raise(self, valid_actions):
        pot_estimate = 15  # Rough preflop pot estimate
        target_bet = pot_estimate * 0.5
        
        if len(valid_actions) > 2:
            raise_info = valid_actions[2]['amount']
            if isinstance(raise_info, dict):
                if raise_info['min'] == -1:
                    return valid_actions[1]['amount']
                return max(min(target_bet, raise_info['max']), raise_info['min'])
            else:
                return raise_info
        return target_bet

    def get_raise_action(self, valid_actions, amount):
        if len(valid_actions) > 2:
            return valid_actions[2]['action'], amount
        else:
            return valid_actions[1]['action'], valid_actions[1]['amount']  

    def should_switch_to_folding(self):
        # Calculate future blind obligations
        if self.is_p1:
            self.total_loss = sum(5 if i % 2 == 1 else 10 for i in range(1, self.remaining_rounds))
        else:
            self.total_loss = sum(10 if i % 2 == 1 else 5 for i in range(1, self.remaining_rounds))
        
        future_stack = self.stack - self.total_loss
        return future_stack > self.emergency_threshold

    def safe_action(self, preferred_action, amount, valid_actions):
        #check
        available_actions = [action['action'] for action in valid_actions]
        
        if preferred_action not in available_actions:
            if 'call' in available_actions:
                preferred_action = 'call'
                amount = valid_actions[1]['amount']
            elif 'fold' in available_actions:
                preferred_action = 'fold'
                amount = 0
            else:
                preferred_action = valid_actions[0]['action']
                amount = valid_actions[0]['amount']
        for action in valid_actions:
            if action['action'] == preferred_action:
                if isinstance(action['amount'], dict):
                    if preferred_action == 'raise':
                        if action['amount']['min'] == -1:
                            preferred_action = 'call'
                            amount = valid_actions[1]['amount']
                            break
                        amount = max(min(amount, action['amount']['max']), action['amount']['min'])
                else:
                    amount = action['amount']
                break

        if preferred_action == 'fold' and self.stack + self.total_loss < 1005:
            preferred_action = 'call'
            amount = valid_actions[1]['amount']

        return preferred_action, amount

    def get_enhanced_state(self, round_state):
        return self.get_state()
    
    def get_state(self):
        return np.random.rand(10)
    
    def get_full_state_analysis(self, round_state):
        state = []
        if self.hole_cards:
            for card in self.hole_cards:
                state.append(self.card_to_rank(card[1]) / 14.0)  # Normalized rank
                state.append(['C', 'D', 'H', 'S'].index(card[0]) / 3.0)  # Normalized suit
        else:
            state.extend([0, 0, 0, 0])
        
        # Community cards (10 features: 5 cards × 2 features each)
        for i in range(5):
            if i < len(self.community_cards):
                card = self.community_cards[i]
                state.extend([
                    self.card_to_rank(card[1]) / 14.0,
                    ['C', 'D', 'H', 'S'].index(card[0]) / 3.0
                ])
            else:
                state.extend([0, 0])
        
        # Game state features (11 features)
        state.extend([
            self.stack / 2000.0,  # Normalized stack
            self.opponent_stack / 2000.0,  # Normalized opponent stack
            self.remaining_rounds / 20.0,  # Normalized remaining rounds
            self.round_count / 20.0,  # Normalized round count
            len(self.community_cards) / 5.0,  # Street indicator
            self.opponent_fold_rate,  # Opponent fold tendency
            self.opponent_raise_rate,  # Opponent raise tendency
            self.estimate_hand_strength(),  # Current hand strength
            self.calculate_pot_size(round_state) / 2000.0,  # Normalized pot size
            (self.opponent_aggressive_count / max(self.opponent_total_actions, 1)),  # Aggression ratio
            1.0 if self.flag else 0.0  # Preflop first action flag
        ])
        
        # ensure 25 dimensions
        while len(state) < 25:
            state.append(0.0)
        
        return np.array(state[:25])

    def calculate_pot_size(self, round_state):
        return sum(seat.get('bet', 0) for seat in round_state['seats'])

    def card_to_rank(self, rank_str):
        rank_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
                   '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        return rank_map.get(rank_str, 2)

    def map_action_index(self, action_idx):
        action_map = ['fold', 'call', 'raise']
        return action_map[action_idx] if action_idx < len(action_map) else 'call'

    def find_action(self, valid_actions, target_action):
        for action in valid_actions:
            if action['action'] == target_action:
                return action['action'], action['amount']
        return valid_actions[0]['action'], valid_actions[0]['amount']

    def log_decision(self, action, amount, round_state):
        decision_info = {
            'round': self.round_count,
            'street': round_state.get('street', 'preflop'),
            'action': action,
            'amount': amount,
            'hand_strength': self.estimate_hand_strength() if self.community_cards else None,
            'stack': self.stack,
            'pot_size': self.calculate_pot_size(round_state)
        }
        self.decision_log.append(decision_info)

    def update_betting_state(self, action, amount, round_state):
        if action in ['raise', 'call']:
            street = round_state.get('street', 'preflop')
            if street not in self.latest_bet:
                self.latest_bet[street] = 0
            self.total_bet -= self.latest_bet[street]
            self.total_bet += amount
            self.latest_bet[street] = amount

    # Game event handlers
    def receive_game_start_message(self, game_info):
        self.hole_cards = []
        self.community_cards = []
        self.total_bet = 0
        self.latest_bet = {}
        self.opponent_actions = []
        self.decision_log = []
        self.flag = False

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.hole_cards = hole_card
        self.community_cards = []
        self.total_bet = 0
        self.latest_bet = {}
        self.flag = False
        self.round_count = round_count

        # Set initial blind obligations
        small_blind_pos = round_count % len(seats)
        big_blind_pos = (round_count + 1) % len(seats)

        for seat in seats:
            if seat['state'] == 'participating' and seat['uuid'] == self.uuid:
                seat_index = seats.index(seat)
                if seat_index == small_blind_pos:
                    self.latest_bet['preflop'] = 5
                elif seat_index == big_blind_pos:
                    self.latest_bet['preflop'] = 10

    def receive_street_start_message(self, street, round_state):
        self.community_cards = round_state['community_card']

    def receive_game_update_message(self, action, round_state):
        self.community_cards = round_state['community_card']
        
        # Track opponent behavior
        if action['player_uuid'] != self.uuid:
            self.update_opponent_model(action['action'])

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

def setup_ai():
    return trainedPlayer()
