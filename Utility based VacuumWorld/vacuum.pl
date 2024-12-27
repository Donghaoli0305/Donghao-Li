:-dynamic square/2.
:-dynamic energyLevel/1.
:-dynamic utility/1.
:-dynamic beenThere/2.
:-dynamic reward/1.
:-dynamic probability/1.
:-dynamic energyuse/1.

%possiblemove(left).
%possiblemove(right).
%possiblemove(back).
%possiblemove(forward).
possiblemove(X) :- random_member(X, [left, right, back, forward]).


% contents of squares that we can't go to.
obstruction(vac). % a bot is there
obstruction(obstacle).

calculateSqrtUtility(Energy, Reward, Probability, Energy_use, Value) :-
						EnergyPlusReward is Energy + Reward,
						((Energy > Energy_use) ->
								EnergyMinusEnergyUse is Energy - Energy_use;
								EnergyMinusEnergyUse is 0),
						write('EnergyMinusEnergyUse is '), write(EnergyMinusEnergyUse), nl,
						Value is Probability * sqrt(EnergyPlusReward) + (1-Probability) * sqrt(EnergyMinusEnergyUse),
						write('Energy is '), write(Energy), nl,
						write('Energy_used is '), write(Energy_use), nl,
						write('Reward is '), write(Reward), nl,
						write('Probability is '), write(Probability), nl,
						write('sqrt(Energy - Energu_use) is )'), write(sqrt(EnergyMinusEnergyUse)), nl,
						write('sqrt(Energy + Reward) is '), write(sqrt(EnergyPlusReward)), nl,
						write('EU is '), write(Value), nl.